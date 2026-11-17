"""
Losses.

The competition metric is IoU on a heavily imbalanced mask (median positive
area 4.8%, first quartile 1.6%).  A single loss is a poor fit:

* plain BCE saturates on the easy vast majority of photographic pixels;
* focal BCE (what v1 used, with a *fixed* ``pos_weight=10``) is even more
  aggressive and, combined with ``alpha=0.25``, silently re-scales the positive
  gradient by 2.5 while the neg/pos ratio varies from 1:3 to 1:10000 between
  images;
* pure Dice/Lovasz are unstable for empty or near-empty masks.

So we use a three-term combination, which is the standard recipe for
segmentation with IoU-like metrics:

    L = w_bce * BCEWithLogits(pos_weight=neg/pos, capped)
      + w_dice * soft Dice (per image)
      + w_lovasz * Lovasz hinge (per image, directly a convex surrogate of IoU)

``pos_weight`` is estimated **per batch** instead of being a constant, and
``Lovasz`` is only applied to the main output (it needs a sort per image).

Everything here is pure PyTorch -- no extra dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BCELoss", "DiceLoss", "LovaszHingeLoss", "ComboLoss",
           "make_criterion"]


# --------------------------------------------------------------------------- #
class BCELoss(nn.Module):
    """BCE with logits and a per-batch, capped positive weight."""

    def __init__(self, max_pos_weight=20.0, adaptive=True, fixed=10.0):
        super().__init__()
        self.max_pos_weight = float(max_pos_weight)
        self.adaptive = bool(adaptive)
        self.fixed = float(fixed)

    def forward(self, logits, target, weight=None):
        if self.adaptive:
            pos = target.sum()
            neg = target.numel() - pos
            pw = (neg / pos.clamp(min=1.0)).clamp(1.0, self.max_pos_weight)
        else:
            pw = torch.as_tensor(self.fixed, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw,
                                                 reduction="none")
        if weight is None:
            return bce.mean()
        # normalise so the magnitude stays comparable to the plain BCE
        return bce.mul(weight).sum() / weight.sum().clamp(min=1e-6)


class DiceLoss(nn.Module):
    """Per-image soft Dice (batch of independent terms, not one global term)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.dim()))
        num = 2.0 * (p * target).sum(dims) + self.smooth
        den = p.sum(dims) + target.sum(dims) + self.smooth
        return 1.0 - (num / den).mean()


class LovaszHingeLoss(nn.Module):
    """Lovasz hinge (Berman et al., CVPR 2018) -- convex IoU surrogate."""

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    @staticmethod
    def _grad(labels_sorted):
        p = labels_sorted.numel()
        gts = labels_sorted.sum()
        intersection = gts - labels_sorted.float().cumsum(0)
        union = gts + (1.0 - labels_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union.clamp(min=1e-9)
        if p > 1:
            jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
        return jaccard

    def _flat(self, logits, labels):
        if labels.numel() == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        grad = self._grad(labels[perm])
        return torch.dot(F.relu(errors_sorted), grad)

    def forward(self, logits, target):
        if self.per_image:
            losses = [self._flat(lg.reshape(-1), tb.reshape(-1))
                      for lg, tb in zip(logits, target)]
            return torch.stack(losses).mean()
        return self._flat(logits.reshape(-1), target.reshape(-1))


class ComboLoss(nn.Module):
    """BCE + Dice + Lovasz on the main output, BCE+Dice on the aux outputs."""

    def __init__(self, w_bce=1.0, w_dice=1.0, w_lovasz=0.5,
                 deep_sup_weights=(1.0, 0.4, 0.15),
                 max_pos_weight=20.0, adaptive_pos_weight=True,
                 boundary_weight=0.0):
        super().__init__()
        self.bce = BCELoss(max_pos_weight=max_pos_weight, adaptive=adaptive_pos_weight)
        self.dice = DiceLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.w_bce, self.w_dice, self.w_lovasz = w_bce, w_dice, w_lovasz
        self.dsw = tuple(deep_sup_weights)
        self.boundary_weight = float(boundary_weight)

    def _boundary_weight_map(self, target):
        """1 + bw on a thin band around the mask edge (IoU is boundary-sensitive)."""
        t = target.float()
        dil = F.max_pool2d(t, 5, 1, 2)
        ero = -F.max_pool2d(-t, 5, 1, 2)
        band = (dil - ero).clamp(0.0, 1.0)
        return 1.0 + self.boundary_weight * band

    def _main(self, logits, target):
        w = self._boundary_weight_map(target) if self.boundary_weight > 0 else None
        loss = self.w_bce * self.bce(logits, target, w) + self.w_dice * self.dice(logits, target)
        if self.w_lovasz > 0:
            loss = loss + self.w_lovasz * self.lovasz(logits, target)
        return loss

    def _aux(self, logits, target):
        t = F.interpolate(target, size=logits.shape[-2:], mode="area")
        return self.w_bce * self.bce(logits, t) + self.w_dice * self.dice(logits, t)

    def forward(self, outputs, target):
        """``outputs`` is the list returned by the model (main first)."""
        if torch.is_tensor(outputs):
            outputs = [outputs]
        loss = self._main(outputs[0], target)
        for k, out in enumerate(outputs[1:3], start=1):
            w = self.dsw[k] if k < len(self.dsw) else 0.0
            if w > 0:
                loss = loss + w * self._aux(out, target)
        return loss


def make_criterion(kind="combo", **kw):
    if kind in ("combo", "bce_dice_lovasz"):
        return ComboLoss(**kw)
    if kind == "bce":
        return ComboLoss(w_dice=0.0, w_lovasz=0.0, **kw)
    if kind == "bce_dice":
        return ComboLoss(w_lovasz=0.0, **kw)
    raise ValueError(f"unknown criterion {kind}")
