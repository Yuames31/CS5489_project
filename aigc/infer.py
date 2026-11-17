"""
Generate the Kaggle submission (``ID,Predicted``).

    python infer.py --model runs/r34_512/model.pth --meta runs/r34_512/model_meta.json \
                    --test-dir "F:\\AIGC数据集\AIGC Detection Dataset 2026\\test\\images" \
                    --out submission.csv

Notes
-----
* RLE is written with the official template encoder (column-major, 1-based
  starts), so the string is directly comparable to ``train.csv``.
* Prediction is tiled at native resolution with overlap and optional flip TTA,
  which is worth a few IoU points over a single squashed forward pass.
* The probability map is thresholded with the value tuned on the validation
  split (stored in ``model_meta.json``); ``--threshold`` overrides it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

import model as model_mod
from data import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD
from inference import predict_prob_map, postprocess_mask
from rle import encode_rle
from util import fix_console

DEFAULT_MODEL = os.path.join("runs", "exp", "model.pth")


def load_model(model_path, meta_path=None, device="cuda"):
    """Load the checkpoint; the meta json is looked up next to the checkpoint."""
    meta = {}
    candidates = [meta_path] if meta_path else []
    candidates.append(os.path.join(os.path.dirname(str(model_path)) or ".", "model_meta.json"))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                meta = json.load(f)
            break
    arch = meta.get("arch", "resnet34")
    net = model_mod.build_model(
        arch, pretrained=False, hf=meta.get("hf", False),
        base=meta.get("base", 32), deep_sup=meta.get("deep_sup", True))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "state_dict" not in state:
        state = state["model"]
    net.load_state_dict(state)
    net.to(device).eval()
    return net, meta


def resolve_checkpoint(model_arg):
    """Allow running with no arguments: find the newest runs/*/model.pth."""
    if model_arg and os.path.isfile(model_arg):
        return model_arg
    if model_arg and model_arg != DEFAULT_MODEL:
        raise SystemExit(f"[infer] checkpoint not found: {model_arg}")
    import glob
    found = sorted(glob.glob(os.path.join("runs", "*", "model.pth")),
                   key=os.path.getmtime, reverse=True)
    if not found:
        raise SystemExit(
            "[infer] no checkpoint found.\n"
            "  pass one explicitly:  --model path\\to\\model.pth\n"
            "  or train first:       python train.py --out-dir runs/exp")
    print(f"[infer] auto-detected checkpoint: {found[0]}")
    return found[0]


def list_test_images(test_dir):
    exts = IMAGE_EXTS
    paths = sorted(p for p in os.listdir(test_dir)
                   if os.path.splitext(p)[1].lower() in exts)
    return [os.path.join(test_dir, p) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser("AIGC inference -> submission.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="checkpoint; if missing under runs/*/model.pth the newest is used")
    ap.add_argument("--meta", default=None,
                    help="model_meta.json (default: next to the checkpoint)")
    ap.add_argument("--test-dir",
                    default=r"F:\AIGC数据集\AIGC Detection Dataset 2026\test\images")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tile", type=int, default=0, help="0 -> from meta / 512")
    ap.add_argument("--stride", type=int, default=0, help="0 -> tile//2")
    ap.add_argument("--tta", type=int, default=4, help="1/2/4 flip-TTA passes")
    ap.add_argument("--scales", default="", help="comma list, e.g. 0.75,1.0")
    ap.add_argument("--threshold", type=float, default=-1.0, help="-1 -> from meta")
    ap.add_argument("--fallback", action="store_true", default=False,
                    help="OFF by default = the exact behaviour of the first "
                         "submitted CSV (0.82). Turning it on rescues images the "
                         "threshold wiped out (+0.6 mean IoU on val).")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false")
    ap.add_argument("--fb-thr", type=float, default=0.45)
    ap.add_argument("--fb-rel", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=0,
                    help="drop components smaller than N pixels (0 = off)")
    ap.add_argument("--fill-holes", action="store_true",
                    help="fill enclosed holes (measured slightly harmful here)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra-models", default="",
                    help="comma-separated extra checkpoints; probability maps are averaged")
    ap.add_argument("--save-probs", default=None, help="dir to dump .npy probability maps")
    a = ap.parse_args(argv)
    fix_console()

    a.model = resolve_checkpoint(a.model)
    net, meta = load_model(a.model, a.meta, a.device)
    extra = [s.strip() for s in a.extra_models.split(",") if s.strip()]
    nets = [net]
    for p in extra:
        nets.append(load_model(p, None, a.device)[0])
    if len(nets) > 1:
        print(f"[infer] ensemble of {len(nets)} checkpoints: "
              f"{[os.path.basename(n) for n in [a.model] + extra]}")
    tile = a.tile or meta.get("val_tile", 512)
    stride = a.stride or meta.get("val_stride", max(tile // 2, 1))
    tta = a.tta if a.tta > 0 else 4
    thr = a.threshold if a.threshold >= 0 else meta.get("threshold", 0.5)
    scales = tuple(float(s) for s in a.scales.split(",")) if a.scales else (1.0,)
    amp_dtype = torch.bfloat16 if (a.device == "cuda" and
                                   torch.cuda.is_bf16_supported()) else None

    paths = list_test_images(a.test_dir)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit(f"no images in {a.test_dir}")
    print(f"[infer] {len(paths)} images | tile={tile} stride={stride} tta={tta} "
          f"scales={scales} thr={thr:.3f} device={a.device}")
    if a.save_probs:
        os.makedirs(a.save_probs, exist_ok=True)

    rows, legacy, t0 = [], [], time.time()
    for i, path in enumerate(paths, 1):
        with Image.open(path) as im:
            im = im.convert("RGB")
            img = np.asarray(im, dtype=np.uint8)
        prob = predict_prob_map(nets[0], img, tile=tile, stride=stride, tta_flips=tta,
                                scales=scales, device=a.device, amp_dtype=amp_dtype)
        for extra_net in nets[1:]:
            prob = prob + predict_prob_map(extra_net, img, tile=tile, stride=stride,
                                           tta_flips=tta, scales=scales,
                                           device=a.device, amp_dtype=amp_dtype)
        if len(nets) > 1:
            prob = prob / len(nets)
        mask = postprocess_mask(
            prob, thr, min_size=a.min_size, fill_holes=a.fill_holes,
            fallback=(a.fb_thr, a.fb_rel) if a.fallback else None).astype(np.uint8)
        if a.save_probs:
            np.save(os.path.join(a.save_probs, f"{os.path.splitext(os.path.basename(path))[0]}.npy"),
                    prob.astype(np.float16))
        img_id = os.path.splitext(os.path.basename(path))[0]
        rle = encode_rle(mask) if mask.any() else ""
        rows.append({"ID": img_id, "Predicted": rle})
        legacy.append({"ID": img_id, "annotation": rle, "height": img.shape[0],
                       "width": img.shape[1], "Usage": "Public"})
        if i % 20 == 0 or i == len(paths):
            el = time.time() - t0
            print(f"  {i}/{len(paths)}  {el / i:.2f}s/img  eta {el / i * (len(paths) - i) / 60:.1f} min",
                  flush=True)

    df = pd.DataFrame(rows, columns=["ID", "Predicted"])
    df.to_csv(a.out, index=False)
    # The official test template expects ``ID,Predicted``, but the packaged
    # ``sample_submission.csv`` follows the training layout.  Write both so
    # either can be uploaded if Kaggle complains about the columns.
    legacy_path = os.path.splitext(a.out)[0] + "_sample_format.csv"
    pd.DataFrame(legacy, columns=["ID", "annotation", "height", "width", "Usage"]).to_csv(
        legacy_path, index=False)
    print(f"[infer] wrote {a.out}  rows={len(df)}  "
          f"non-empty={int((df.Predicted.str.len() > 0).sum())}")
    print(f"[infer] wrote {legacy_path}  (ID,annotation,height,width,Usage)")
    return 0


# ============================================================================
# ============================================================================
# PRESET_ARGS -- arguments used when you press "Run" in PyCharm.
# Edit only this list; [] = argparse defaults. Examples:
#   evaluate: ["--limit","200","--tta","1"]
#             ["--limit","200","--tta","1","--scales","1.0,1.5"]
#             ["--model","runs/r3_512/model.pth","--tta","1"]
#   ensemble: ["--runs","runs/exp","runs/r3_512","--eval","--tta","2","--limit","250"]
#             ["--runs","runs/exp","runs/r3_512","--weights","auto","--tta","4","--out","submission_final.csv"]
# ============================================================================
PRESET_ARGS = []



if __name__ == "__main__":
    # force the working directory to this folder so relative paths are stable
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main(PRESET_ARGS if len(sys.argv) <= 1 else None))
