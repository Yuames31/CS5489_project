"""
RLE codec for the CS5489 AIGC-Detection dataset.

Ground truth (verified empirically against the released `train.csv`):

* scan order  : **column-major** (Fortran order), i.e. `mask.ravel(order="F")`
* indexing    : **1-based** start positions
* layout      : flat list of ``start length start length ...`` pairs
* empty mask  : empty string

Evidence:
  * the smallest start position over the whole 5000-image train set is 1,
    no annotation contains a 0, and 115/5000 rows would overflow ``H*W``
    if interpreted as 0-based, while 0/5000 overflow when read as 1-based;
  * consecutive runs advance by ~H (median |Δ - H| = 2) rather than ~W
    (median |Δ - W| = 512), which happens only when a run walks *down a
    column* => column-major;
  * overlaying the decoded mask on the image gives object-shaped regions
    that follow the edited objects exactly (see docs/figures).

`encode_rle` is a byte-for-byte copy of the function that ships with the
official test template and must not be changed.

All decoders below are vectorised and never materialise the full-size mask
unless ``decode_full`` is used, which keeps patch training cheap even for
16-megapixel images.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rle_to_arrays",
    "decode_full",
    "decode_window",
    "positive_pixels",
    "count_positive",
    "encode_rle",
]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def rle_to_arrays(rle, dtype=np.int64):
    """Parse an RLE string into 0-based ``(start, length)`` arrays."""
    s = str(rle).split() if rle is not None else []
    if len(s) == 0:
        empty = np.zeros(0, dtype=dtype)
        return empty, empty
    if len(s) % 2 != 0:
        raise ValueError(f"RLE string has an odd number of tokens: {len(s)}")
    a = np.asarray(s, dtype=dtype)
    starts = a[0::2] - 1   # dataset is 1-based -> 0-based
    lengths = a[1::2]
    return starts, lengths


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _runs_to_mask(starts, lengths, n_total):
    """Vectorised run -> flat boolean mask (length ``n_total``)."""
    if starts.size == 0:
        return np.zeros(n_total, dtype=bool)
    s = np.clip(starts, 0, n_total)
    e = np.clip(starts + lengths, 0, n_total)
    keep = s < e
    if not keep.any():
        return np.zeros(n_total, dtype=bool)
    s, e = s[keep], e[keep]
    diff = np.zeros(n_total + 1, dtype=np.int32)
    np.add.at(diff, s, 1)
    np.add.at(diff, e, -1)
    np.cumsum(diff, out=diff)          # in place: half the peak memory
    return diff[:-1] > 0


def decode_full(rle, height: int, width: int, out: np.ndarray | None = None) -> np.ndarray:
    """Decode an RLE string into a ``(height, width)`` uint8 mask."""
    starts, lengths = rle_to_arrays(rle)
    flat = _runs_to_mask(starts, lengths, height * width)
    mask = flat.reshape((height, width), order="F")
    if out is None:
        return mask.astype(np.uint8, copy=False)
    out[...] = mask
    return out


def decode_window(rle, height: int, width: int,
                  r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Decode only the window ``rows [r0, r1) x cols [c0, c1)``.

    Column-major order means a range of columns is contiguous in flat index
    space (``[c0*H, c1*H)``), so the window can be reconstructed without ever
    touching the other columns.  Cost is O(window pixels + runs).
    """
    r0, r1 = max(0, int(r0)), min(int(height), int(r1))
    c0, c1 = max(0, int(c0)), min(int(width), int(c1))
    win_h, win_w = r1 - r0, c1 - c0
    if win_h <= 0 or win_w <= 0:
        return np.zeros((max(win_h, 0), max(win_w, 0)), dtype=np.uint8)

    starts, lengths = rle_to_arrays(rle)
    if starts.size == 0:
        return np.zeros((win_h, win_w), dtype=np.uint8)

    a, b = c0 * height, c1 * height                      # flat range of window
    ends = starts + lengths
    s = np.maximum(starts, a)
    e = np.minimum(ends, b)
    keep = s < e
    if not keep.any():
        return np.zeros((win_h, win_w), dtype=np.uint8)
    s = (s[keep] - a).astype(np.int64)
    e = (e[keep] - a).astype(np.int64)

    flat = _runs_to_mask(s, e - s, b - a).reshape((height, win_w), order="F")
    return flat[r0:r1].astype(np.uint8, copy=False)


# --------------------------------------------------------------------------- #
# stats / sampling helpers
# --------------------------------------------------------------------------- #
def count_positive(rle, height: int, width: int) -> int:
    """Number of positive pixels (sum of run lengths)."""
    _, lengths = rle_to_arrays(rle)
    return int(lengths.sum())


def positive_pixels(rle, height: int, width: int, n: int, rng) -> np.ndarray:
    """Sample ``n`` positive pixel coordinates as ``(n, 2)`` ``[row, col]``.

    Runs are sorted, so we sample uniformly over the flat positive area with
    one search per sample; cost is O(n log #runs) and memory O(1).
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    starts, lengths = rle_to_arrays(rle)
    if starts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    cum = np.cumsum(lengths)
    total = int(cum[-1])
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64)
    offsets = rng.integers(0, total, size=n)
    idx = np.searchsorted(cum, offsets, side="right")
    prev = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0)
    flat = starts[idx] + (offsets - prev)
    rows = flat % height
    cols = flat // height
    return np.stack([rows, cols], axis=1).astype(np.int64)


# --------------------------------------------------------------------------- #
# encoding (identical to the official template -- do not modify)
# --------------------------------------------------------------------------- #
def encode_rle(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Predicted mask must be 2D and contain only 0/1")
    pixels = mask.astype(np.uint8).ravel(order="F")
    padded = np.pad(pixels, (1, 1), constant_values=0)
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[::2] + 1
    lengths = changes[1::2] - changes[::2]
    return " ".join(str(v) for pair in zip(starts, lengths) for v in pair)
