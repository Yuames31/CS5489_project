"""
Metrics.

The leaderboard is IoU based, so we track both the two standard variants:

* ``mean_iou``  : mean over images of per-image IoU  (usually what the
  leaderboard reports, and what we select checkpoints on);
* ``global_iou``: total intersection / total union (dominated by big masks).

Dice and pixel accuracy are kept for the report.  Stats are accumulated for
several thresholds at once so a threshold sweep costs almost nothing.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SegMetrics", "iou_dice_from_stats"]


def iou_dice_from_stats(tp, fp, fn):
    """IoU / Dice from integer counts; empty-vs-empty counts as a perfect match."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    union = tp + fp + fn
    iou = np.where(union > 0, tp / np.maximum(union, 1e-12), 1.0)
    dice_den = 2.0 * tp + fp + fn
    dice = np.where(dice_den > 0, 2.0 * tp / np.maximum(dice_den, 1e-12), 1.0)
    return iou, dice


class SegMetrics:
    """Accumulates per-image confusion counts across thresholds."""

    def __init__(self, thresholds=(0.5,)):
        self.thresholds = tuple(float(t) for t in thresholds)
        self.reset()

    def reset(self):
        n = len(self.thresholds)
        self.tp = [np.zeros(0) for _ in range(n)]
        self.fp = [np.zeros(0) for _ in range(n)]
        self.fn = [np.zeros(0) for _ in range(n)]
        self.tn = [np.zeros(0) for _ in range(n)]
        self.ids = []
        self.gt_area = []

    def update(self, prob, target, image_id="-"):
        prob = np.asarray(prob)
        target = np.asarray(target).astype(bool)
        self.ids.append(str(image_id))
        self.gt_area.append(int(target.sum()))
        for k, thr in enumerate(self.thresholds):
            pred = prob >= thr
            tp = np.count_nonzero(pred & target)
            fp = np.count_nonzero(pred & ~target)
            fn = np.count_nonzero(~pred & target)
            tn = np.count_nonzero(~pred & ~target)
            self.tp[k] = np.append(self.tp[k], tp)
            self.fp[k] = np.append(self.fp[k], fp)
            self.fn[k] = np.append(self.fn[k], fn)
            self.tn[k] = np.append(self.tn[k], tn)

    # ------------------------------------------------------------------ report
    def summary(self):
        """Dict keyed by threshold -> metric dict (floats)."""
        out = {}
        for k, thr in enumerate(self.thresholds):
            tp, fp, fn, tn = self.tp[k], self.fp[k], self.fn[k], self.tn[k]
            iou, dice = iou_dice_from_stats(tp, fp, fn)
            total = tp.sum() + fp.sum() + fn.sum() + tn.sum()
            out[thr] = {
                "thr": thr,
                "mean_iou": float(iou.mean()) if len(iou) else 0.0,
                "median_iou": float(np.median(iou)) if len(iou) else 0.0,
                "global_iou": float(tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1)),
                "mean_dice": float(dice.mean()) if len(dice) else 0.0,
                "pixel_acc": float((tp.sum() + tn.sum()) / max(total, 1)),
                "precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
                "recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
                "n_images": int(len(iou)),
            }
        return out

    def best_threshold(self, key="mean_iou"):
        s = self.summary()
        thr = max(s, key=lambda t: s[t][key])
        return thr, s[thr]

    def per_image_table(self, thr):
        k = self.thresholds.index(float(thr))
        iou, dice = iou_dice_from_stats(self.tp[k], self.fp[k], self.fn[k])
        rows = []
        for i in range(len(self.ids)):
            total = self.tp[k][i] + self.fp[k][i] + self.fn[k][i] + self.tn[k][i]
            rows.append({
                "ID": self.ids[i], "iou": float(iou[i]), "dice": float(dice[i]),
                "gt_pixels": int(self.gt_area[i]),
                "pred_pixels": int(self.tp[k][i] + self.fp[k][i]),
                "gt_ratio": float(self.gt_area[i]) / max(total, 1),
                "tp": int(self.tp[k][i]), "fp": int(self.fp[k][i]),
                "fn": int(self.fn[k][i]),
            })
        return rows
