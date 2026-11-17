"""
Training entry point.

    python train.py --out-dir runs/r34_512 --arch resnet34 --crop 512 --epochs 30

What changed w.r.t. the v1 baseline (see README.md for the full list):

1. native-resolution crops instead of a 256x256 resize of the whole image;
2. a real train/val split -- in v1 ``val_set.dataset.is_train = False`` was
   applied to the *shared* dataset object, which silently switched augmentation
   off for the training set too;
3. BCE(pos_weight=neg/pos) + Dice + Lovasz instead of Focal + Dice;
4. deep supervision, SCSE attention, full-resolution refinement branch;
5. AdamW + warm-up + cosine, gradient clipping, AMP, EMA weights;
6. per-epoch validation with tiled full-resolution inference, a threshold
   sweep, and both mean and global IoU.

Every run writes ``model.pth`` (plain state_dict, EMA weights, what the TA
notebook loads), ``model_meta.json`` (architecture + preprocessing + tuned
threshold) and ``history.csv`` (for the report figures).
"""
from __future__ import annotations

import argparse
import copy
import csv as _csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data as data_mod
import losses
import model as model_mod
from inference import predict_prob_map
from metrics import SegMetrics
from util import fix_console

HISTORY_FIELDS = ["epoch", "train_loss", "lr", "val_iou", "val_iou_global",
                  "val_dice", "val_thr", "val_iou_at_05", "sec"]


# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser("AIGC dense-prediction training", conflict_handler="resolve")
    p.add_argument("--data-root", default=r"F:\AIGC数据集\AIGC Detection Dataset 2026")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--out-dir", default="runs/exp")

    # model
    p.add_argument("--arch", default="resnet34",
                   choices=["resnet18", "resnet34", "resnet50", "deeplabv3"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--hf", action="store_true", help="feed a high-pass residual into the stem")
    p.add_argument("--base", type=int, default=32, help="decoder width")
    p.add_argument("--no-deep-sup", action="store_true")

    # data
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--scale-min", type=float, default=0.5)
    p.add_argument("--scale-max", type=float, default=1.5)
    p.add_argument("--pos-prob", type=float, default=0.7)
    p.add_argument("--gamma-aug", action="store_true",
                   help="random gamma 0.6-1.6 (helps low-light / flat edits)")
    p.add_argument("--area-balanced", action="store_true",
                   help="oversample images whose mask is small")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)

    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4, help="decoder / head lr")
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--loss", default="combo",
                   choices=["combo", "bce", "bce_dice", "bce_dice_lovasz"])
    p.add_argument("--w-bce", type=float, default=1.0)
    p.add_argument("--w-dice", type=float, default=1.0)
    p.add_argument("--w-lovasz", type=float, default=0.5)
    p.add_argument("--max-pos-weight", type=float, default=20.0)
    p.add_argument("--boundary-weight", type=float, default=0.0,
                   help="extra BCE weight on the mask boundary band, e.g. 2.0")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--patience", type=int, default=10, help="early stop epochs")

    # validation
    p.add_argument("--val-images", type=int, default=200,
                   help="images used for the per-epoch model selection")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-tile", type=int, default=512)
    p.add_argument("--val-stride", type=int, default=384)
    p.add_argument("--val-tta", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--val-thresholds", default="0.3,0.4,0.5,0.6,0.7")

    # misc
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="synthetic data, 2 steps -- validates the code path only")
    p.add_argument("--smoke", action="store_true",
                   help="tiny real run (4 train / 2 val images, 1 epoch)")
    return p.parse_args(argv)


def finalise_args(a):
    if a.train_csv is None:
        a.train_csv = os.path.join(a.data_root, "train.csv")
    if a.img_dir is None:
        a.img_dir = os.path.join(a.data_root, "train", "images")
    if a.smoke:
        a.limit_train, a.limit_val = 4, 2
        a.epochs, a.crop, a.batch_size, a.workers = 1, 256, 1, 0
        a.val_images, a.val_tile, a.val_stride = 2, 256, 192
        a.device = "cpu"
    if a.device == "cpu":
        a.amp = "none"
    return a


# --------------------------------------------------------------------------- #
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach().float(), alpha=1.0 - d)
            else:
                e.copy_(m)


def build_optimizer(model, a):
    enc, dec = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith(("stem", "layer", "maxpool")) else dec).append((n, p))
    groups = [
        {"params": [p for _, p in enc], "lr": a.encoder_lr, "name": "encoder"},
        {"params": [p for _, p in dec], "lr": a.lr, "name": "decoder"},
    ]
    return torch.optim.AdamW(groups, weight_decay=a.weight_decay)


def build_scheduler(opt, a, steps_per_epoch):
    total = max(a.epochs * steps_per_epoch, 1)
    warm = max(int(a.warmup_epochs * steps_per_epoch), 1)

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(total - warm, 1)
        return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def amp_dtype_of(a):
    if a.amp == "none" or not torch.cuda.is_available():
        return None
    if a.amp == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float16


def _make_scaler(dt):
    """GradScaler only for fp16 (bf16 needs none)."""
    if dt != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, opt, sched, criterion, device, a, scaler, ema, epoch):
    model.train()
    if hasattr(loader.dataset, "epoch"):
        loader.dataset.epoch = epoch
    dt = amp_dtype_of(a)
    running, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    t0 = time.time()

    for i, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=dt) if dt is not None else _nullctx():
            out = model(x)
            loss = criterion(out, y)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % a.accum == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if ema is not None:
                ema.update(model)

        running += float(loss.detach())
        n += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            ips = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"    iter {i + 1}/{len(loader)}  loss {running / n:.4f}  {ips:.2f} it/s",
                  flush=True)

    return running / max(n, 1)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(records, img_dir, model, device, a, thresholds):
    model.eval()
    ds = data_mod.AIGCFullDataset(records, img_dir)
    # decoding a full-resolution mask costs ~130 MB per worker in the worst case,
    # so keep the validation loader small
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=min(a.workers, 2),
                        collate_fn=data_mod.collate_full,
                        pin_memory=(a.device != "cpu"))
    n_total = len(loader)
    metrics = SegMetrics(thresholds=thresholds)
    dt = amp_dtype_of(a)
    t0 = time.time()
    for i, item in enumerate(loader):
        img = item["image"].numpy().transpose(1, 2, 0).astype(np.uint8)
        prob = predict_prob_map(model, img, tile=a.val_tile, stride=a.val_stride,
                                tta_flips=a.val_tta, scales=(1.0,),
                                device=device, amp_dtype=dt)
        metrics.update(prob, item["mask"].numpy(), item["id"])
        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            print(f"    val {i + 1}/{n_total}  ({(time.time() - t0) / (i + 1):.2f}s/img)",
                  flush=True)
    return metrics


# --------------------------------------------------------------------------- #
def save_checkpoint(path, state_dict):
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, path)


def main(argv=None):
    fix_console()
    a = finalise_args(get_args(argv))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    thresholds = [float(t) for t in a.val_thresholds.split(",")]
    print(f"[cfg] {json.dumps({k: v for k, v in vars(a).items() if k != 'device'}, ensure_ascii=False)}")

    # ------------------------------------------------------------------ data
    if a.dry_run:
        class _Synth(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, i):
                return {"image": torch.randn(3, a.crop, a.crop),
                        "mask": (torch.rand(1, a.crop, a.crop) > 0.9).float(),
                        "id": f"s{i}", "orig_hw": torch.tensor([a.crop, a.crop])}

        train_set = _Synth()
        val_records = [{"id": f"v{i}", "rle": "", "h": a.crop, "w": a.crop, "n_pos": 0}
                       for i in range(2)]
        train_list = val_list = train_set
    else:
        records = data_mod.read_manifest(a.train_csv)
        train_records, val_records = data_mod.make_split(records, a.val_ratio, a.seed)
        # deterministic subset of the validation split for per-epoch selection
        order = np.random.RandomState(a.seed + 1).permutation(len(val_records))
        val_records = [val_records[i] for i in order]
        if a.limit_train:
            train_records = train_records[:a.limit_train]
        n_val_each = a.val_images if a.val_images > 0 else len(val_records)
        if a.limit_val:
            n_val_each = min(n_val_each, a.limit_val)
        val_records = val_records[:n_val_each]

        train_set = data_mod.AIGCPatchDataset(
            train_records, a.img_dir, crop=a.crop,
            scale_range=(a.scale_min, a.scale_max), pos_prob=a.pos_prob,
            augment=True, seed=a.seed, gamma_aug=a.gamma_aug)
        train_list = train_records
        val_list = val_records

    sampler = None
    if a.area_balanced and not a.dry_run:
        import numpy as _np
        w = data_mod.area_sampling_weights(train_records)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(train_records),
            replacement=True)
        print(f"[data] area-balanced sampling ON "
              f"(weight range {w.min():.2f}~{w.max():.2f}, mean {w.mean():.2f})")

    train_loader = DataLoader(
        train_set, batch_size=a.batch_size, shuffle=(sampler is None and not a.dry_run),
        sampler=sampler,
        num_workers=a.workers, pin_memory=(a.device != "cpu"),
        drop_last=False, persistent_workers=False)
    print(f"[data] train={len(train_list)}  val(eval)={len(val_list)}  "
          f"crop={a.crop}  batch={a.batch_size}x{a.accum}")

    # ----------------------------------------------------------------- model
    model = model_mod.build_model(
        a.arch, pretrained=not a.no_pretrained, hf=a.hf, base=a.base,
        deep_sup=not a.no_deep_sup).to(a.device)
    print(f"[model] {a.arch} params={model_mod.count_params(model) / 1e6:.2f}M  "
          f"hf={a.hf} device={a.device}")

    criterion = losses.make_criterion(
        a.loss, w_bce=a.w_bce, w_dice=a.w_dice, w_lovasz=a.w_lovasz,
        max_pos_weight=a.max_pos_weight, boundary_weight=a.boundary_weight)
    opt = build_optimizer(model, a)
    steps = max(len(train_loader) // max(a.accum, 1), 1)
    sched = build_scheduler(opt, a, steps)
    dt = amp_dtype_of(a)
    scaler = _make_scaler(dt)
    ema = ModelEMA(model, a.ema_decay) if a.ema_decay > 0 else None

    start_epoch, best = 1, -1.0
    if a.resume:
        ckpt = torch.load(a.resume, map_location=a.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best", -1.0)
        print(f"[resume] from {a.resume} at epoch {start_epoch}")

    history = []
    hist_path = os.path.join(a.out_dir, "history.csv")
    if os.path.exists(hist_path) and not a.resume:
        os.remove(hist_path)

    max_epochs = 2 if a.dry_run else a.epochs
    bad_epochs = 0
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        if a.dry_run:
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                x = batch["image"].to(a.device)
                y = batch["mask"].to(a.device)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                if ema is not None:
                    ema.update(model)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                train_loss += float(loss)
                if i >= 1:
                    break
            print(f"[dry-run] loss={train_loss:.4f} -> forward/backward/step/EMA OK")
            return 0

        train_loss = train_one_epoch(model, train_loader, opt, sched, criterion,
                                     a.device, a, scaler, ema, epoch)

        row = {"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[1]["lr"],
               "sec": round(time.time() - t0, 1)}

        if epoch % a.val_every == 0 or epoch == max_epochs or epoch == 1:
            eval_model = ema.ema if ema is not None else model
            metrics = validate(val_list, a.img_dir, eval_model, a.device, a, thresholds)
            summ = metrics.summary()
            best_thr, best_metrics = metrics.best_threshold("mean_iou")
            row.update({
                "val_iou": best_metrics["mean_iou"],
                "val_iou_global": best_metrics["global_iou"],
                "val_dice": best_metrics["mean_dice"],
                "val_thr": best_thr,
                "val_iou_at_05": summ[0.5]["mean_iou"] if 0.5 in summ else float("nan"),
                "sec": round(time.time() - t0, 1),
            })
            print(f"[epoch {epoch:02d}] train {train_loss:.4f} | "
                  f"val IoU {best_metrics['mean_iou']:.4f} (thr {best_thr:.2f}) | "
                  f"global IoU {best_metrics['global_iou']:.4f} | "
                  f"Dice {best_metrics['mean_dice']:.4f} | {row['sec']:.0f}s", flush=True)

            if best_metrics["mean_iou"] > best:
                best = best_metrics["mean_iou"]
                bad_epochs = 0
                sd = (ema.ema if ema is not None else model).state_dict()
                save_checkpoint(os.path.join(a.out_dir, "model.pth"), sd)
                meta = {
                    "arch": a.arch, "hf": a.hf, "base": a.base,
                    "deep_sup": not a.no_deep_sup,
                    "crop": a.crop, "norm_mean": list(data_mod.IMAGENET_MEAN),
                    "norm_std": list(data_mod.IMAGENET_STD),
                    "threshold": float(best_thr),
                    "val_tile": a.val_tile, "val_stride": a.val_stride,
                    "val_tta": a.val_tta,
                    "epoch": epoch, "val": best_metrics,
                    "val_images": len(val_list),
                    "train_images": len(train_list),
                }
                with open(os.path.join(a.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"    -> saved model.pth (IoU {best:.4f})")
            else:
                bad_epochs += 1

        else:
            print(f"[epoch {epoch:02d}] train {train_loss:.4f} | {row['sec']:.0f}s", flush=True)

        history.append(row)
        write_header = not os.path.exists(hist_path)
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

        torch.save({"model": model.state_dict(),
                    "ema": ema.ema.state_dict() if ema else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best,
                    "args": vars(a)}, os.path.join(a.out_dir, "last.pth"))

        if a.patience and bad_epochs >= a.patience:
            print(f"[early-stop] no improvement for {bad_epochs} epochs")
            break

    print(f"[done] best mean IoU {best:.4f}  ->  {os.path.join(a.out_dir, 'model.pth')}")
    return 0

# ============================================================================
# PRESET_ARGS -- the arguments used when you press "Run" in PyCharm.
# Edit only this list; no command line needed.  [] = argparse defaults.
# NOTE: keep --out-dir pointing at a NEW folder, otherwise the model that
# produced submission_v2 (runs/exp/model.pth) gets overwritten.
# The config that produced runs/exp (val mean IoU 0.8120):
#   ["--out-dir","runs/exp_retrain","--crop","512","--batch-size","8","--epochs","30"]
# ============================================================================
PRESET_ARGS = ["--out-dir", "runs/exp_retrain"]


if __name__ == "__main__":
    # force the working directory to this folder so relative paths are stable
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main(PRESET_ARGS if len(sys.argv) <= 1 else None))
