"""
Inference: sliding-window (tiled) full-resolution prediction + TTA.

Predicting by squashing a 2448x2448 image into 256x256 (v1) is the single worst
thing one can do here -- it both destroys the high-frequency forgery evidence
and blurs the mask outline.  Predicting a whole 4000x4000 image in one forward
pass does not fit in 6-8 GB either.

So we run the network on 512x512 tiles at *native* resolution with a 25%
overlap, blend with a Hann window (no visible seams), and optionally average
over flips / scales (TTA).  The probability map is returned at the original
resolution and thresholded afterwards, so the threshold can be tuned on the
validation split without re-running the network.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["predict_prob_map", "normalize_chw", "tile_starts", "hann2d",
           "postprocess_mask"]


# --------------------------------------------------------------------------- #
def postprocess_mask(prob, thr, min_size=0, fill_holes=False, fallback=None):
    """Probability map -> binary mask, plus the three cheap post-processing knobs.

    Tuned on the 500-image validation split (mean IoU 0.8120 -> 0.8181):

    * ``fallback=(0.45, 0.5)`` -- if the threshold wipes an image out completely
      while the maximum probability is still >= 0.45, re-threshold that image
      relatively.  **All of the measured gain came from this rule**: 21 images
      had IoU = 0 simply because no pixel exceeded 0.7.
    * ``min_size`` -- drop connected components smaller than N pixels.  Measured
      effect +0.0003 (noise); kept for completeness, default 0.
    * ``fill_holes`` -- fill enclosed holes.  Measured effect was *negative*
      (-0.0034) on this dataset, default off.
    """
    mask = prob >= thr
    if fallback is not None and not mask.any():
        fb_thr, fb_rel = fallback
        pmax = float(prob.max())
        if pmax >= fb_thr:
            mask = prob >= max(0.25 * thr, fb_rel * pmax)
    if mask.any() and (fill_holes or min_size > 0):
        from scipy import ndimage
        if fill_holes:
            mask = ndimage.binary_fill_holes(mask)
        if min_size > 0:
            lab, n = ndimage.label(mask)
            if n:
                sizes = np.bincount(lab.ravel(), minlength=n + 1)
                keep = sizes >= min_size
                keep[0] = False
                mask = keep[lab] if keep.any() else np.zeros_like(mask)
    return mask


# --------------------------------------------------------------------------- #
def normalize_chw(img_uint8_hwc: np.ndarray) -> np.ndarray:
    x = img_uint8_hwc.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def hann2d(tile: int) -> np.ndarray:
    w = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win = np.outer(w, w)
    return np.clip(win, 1e-2, None)


def tile_starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(stride, 1)))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_tiles(model, img_chw: torch.Tensor, tile: int, stride: int,
                   device, amp_dtype=None):
    """img_chw: (3,h,w) float32 on CPU. Returns (prob, weight) at (h,w)."""
    _, h, w = img_chw.shape
    pad_h = max(tile - h, 0)
    pad_w = max(tile - w, 0)
    if pad_h or pad_w:
        img_chw = F.pad(img_chw[None], (0, pad_w, 0, pad_h), mode="replicate")[0]
    H, W = img_chw.shape[-2:]

    win = torch.from_numpy(hann2d(tile))
    acc = torch.zeros(H, W, dtype=torch.float32, device=device)
    wsum = torch.zeros(H, W, dtype=torch.float32, device=device)

    x = img_chw.to(device, non_blocking=True)
    win = win.to(device)

    ys = tile_starts(H, tile, stride)
    xs = tile_starts(W, tile, stride)
    for y in ys:
        for x0 in xs:
            patch = x[:, y:y + tile, x0:x0 + tile][None]
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(patch)[0]
            else:
                logits = model(patch)[0]
            prob = torch.sigmoid(logits.float())[0, 0]
            acc[y:y + tile, x0:x0 + tile] += prob * win
            wsum[y:y + tile, x0:x0 + tile] += win
    prob = acc / wsum.clamp(min=1e-6)
    return prob[:h, :w].cpu()


@torch.no_grad()
def predict_prob_map(model, image, tile=512, stride=None, tta_flips=2,
                     scales=(1.0,), device="cuda", amp_dtype=None):
    """Full-resolution probability map for one RGB image.

    Args:
        image: ``PIL.Image`` or ``np.ndarray`` (H, W, 3) uint8, RGB.
        tile:  network input size (must match training crop, or a multiple).
        stride: tile stride, default ``tile // 2`` (50% overlap).
        tta_flips: 1 = identity, 2 = +hflip, 4 = full dihedral flips.
        scales:  list of resize factors, e.g. ``(1.0,)`` or ``(0.75, 1.0)``.
    Returns:
        np.ndarray (H, W) float32 probabilities in [0, 1].
    """
    if isinstance(image, Image.Image):
        img = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        img = np.asarray(image, dtype=np.uint8)
    H, W = img.shape[:2]
    stride = stride or max(tile // 2, 1)

    flips = [(False, False), (True, False), (False, True), (True, True)][:max(1, tta_flips)]
    acc = torch.zeros(H, W, dtype=torch.float32)
    n = 0

    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img_s = img
        else:
            nw, nh = max(int(round(W * s)), 8), max(int(round(H * s)), 8)
            img_s = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        x = torch.from_numpy(normalize_chw(img_s))
        for fh, fv in flips:
            xs = x
            if fh:
                xs = torch.flip(xs, dims=[2])
            if fv:
                xs = torch.flip(xs, dims=[1])
            p = _forward_tiles(model, xs, tile, stride, device, amp_dtype)
            if fh:
                p = torch.flip(p, dims=[1])
            if fv:
                p = torch.flip(p, dims=[0])
            if p.shape != (H, W):
                p = F.interpolate(p[None, None], size=(H, W), mode="bilinear",
                                  align_corners=False)[0, 0]
            acc += p
            n += 1

    return (acc / max(n, 1)).numpy()
