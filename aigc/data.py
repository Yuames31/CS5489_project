"""
Datasets.

Key design decision (this is the main fix over the v1 baseline): the images are
*not* resized to a small square.  AIGC forgeries live in high-frequency image
statistics (GAN/diffusion up-sampling artefacts, noise residual mismatch), and
the median image here is 1523x1600 -- squashing it to 256x256 destroys exactly
the signal the task is about, and also destroys thin mask structures.

Instead we train on **native-resolution random crops**:

    pick a window of ``crop/scale`` native pixels (scale in [0.5, 1.5])
    -> resize that window to ``crop x crop`` -> random flips / rot90

so the network always sees roughly ``crop`` pixels of context at roughly the
original sampling rate, and the mask is decoded *only inside the window*
(see ``rle.decode_window``), which keeps a 16 Mpx image cheap to sample.

Windows are sampled positively-biased (``pos_prob``): the positive area is on
average 11.6% of an image but the median is 4.8%, so uniform crops would waste
most of the gradient on all-negative tiles.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

import rle

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
def read_manifest(csv_path: str):
    """Return a list of ``dict(id, rle, h, w, n_pos)`` records."""
    df = pd.read_csv(csv_path, dtype={"ID": str})
    cols = {c.lower(): c for c in df.columns}
    id_col = cols["id"]
    rle_col = cols.get("annotation") or cols.get("predicted")
    h_col, w_col = cols.get("height"), cols.get("width")

    records = []
    for i in range(len(df)):
        rid = str(df[id_col].iloc[i])
        ann = df[rle_col].iloc[i] if rle_col is not None else ""
        ann = "" if (pd.isna(ann) or str(ann).lower() in ("nan", "none")) else str(ann)
        if h_col is not None and w_col is not None and not pd.isna(df[h_col].iloc[i]):
            h, w = int(df[h_col].iloc[i]), int(df[w_col].iloc[i])
        else:                                  # fall back to the image header
            with Image.open(find_image("", rid)) as im:
                w, h = im.size
        n_pos = rle.count_positive(ann, h, w) if ann else 0
        records.append(dict(id=rid, rle=ann, h=h, w=w, n_pos=n_pos))
    return records


def find_image(img_dir: str, img_id: str) -> str:
    for ext in IMAGE_EXTS:
        p = os.path.join(img_dir, f"{img_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no image for id {img_id} in {img_dir}")


def make_split(records, val_ratio=0.1, seed=42):
    """Deterministic id-based split (stable across runs and machines)."""
    ids = sorted(r["id"] for r in records)
    perm = np.random.RandomState(seed).permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_ids = {ids[i] for i in perm[:n_val]}
    train = [r for r in records if r["id"] not in val_ids]
    val = [r for r in records if r["id"] in val_ids]
    return train, val


def area_sampling_weights(records, edges=(0.01, 0.05, 0.20, 0.50), max_ratio=3.0):
    """Per-sample weights that even out the mask-size buckets.

    Small edits are the weakest bucket (IoU 0.69 vs 0.92 for 20-50% masks) but
    only 16% of the images, so they get ~1/4 of the gradient.  Inverse-frequency
    weights (softened with a square root, clipped to ``max_ratio``) give them a
    larger share without turning the batch into only-small-masks.
    """
    ratio = np.array([r["n_pos"] / max(r["h"] * r["w"], 1) for r in records], dtype=np.float64)
    bucket = np.digitize(ratio, edges)
    counts = np.bincount(bucket, minlength=len(edges) + 1).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts[bucket], 1.0))
    w = w / w.mean()
    return np.clip(w, 1.0 / max_ratio, max_ratio)


# --------------------------------------------------------------------------- #
def _to_tensor(img_chw_uint8: np.ndarray, normalize=True) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img_chw_uint8)).float() / 255.0
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
    return x


class AIGCPatchDataset(Dataset):
    """Native-resolution patch dataset (training / fast validation)."""

    def __init__(self, records, img_dir: str, crop: int = 512,
                 scale_range=(0.5, 1.5), pos_prob: float = 0.7,
                 augment: bool = True, seed: int = 0, epoch: int = 0,
                 jitter: float = 0.25, noise_sigma: float = 4.0,
                 gamma_aug: bool = False):
        self.records = records
        self.img_dir = img_dir
        self.crop = int(crop)
        self.scale_range = tuple(scale_range)
        self.pos_prob = float(pos_prob)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.jitter = float(jitter)
        self.noise_sigma = float(noise_sigma)
        self.gamma_aug = bool(gamma_aug)

    def __len__(self):
        return len(self.records)

    # ---------------------------------------------------------------- helpers
    def _rng(self, idx):
        # deterministic in (seed, epoch, idx) so runs are reproducible while
        # still changing every epoch
        s = (self.seed * 1_000_003 + self.epoch * 917_513 + idx * 2_654_435_761) % (2**63 - 1)
        return np.random.default_rng(s)

    def _window(self, rec, rng):
        H, W = rec["h"], rec["w"]
        scale = rng.uniform(*self.scale_range) if self.augment else 1.0
        ws = int(round(self.crop / max(scale, 1e-6)))
        ws = int(np.clip(ws, 32, min(H, W)))

        use_pos = self.augment and rec["n_pos"] > 0 and rng.random() < self.pos_prob
        if use_pos:
            pts = rle.positive_pixels(rec["rle"], H, W, 1, rng)
            cy, cx = int(pts[0, 0]), int(pts[0, 1])
            j = int(round(ws * self.jitter))
            top = cy - ws // 2 + int(rng.integers(-j, j + 1))
            left = cx - ws // 2 + int(rng.integers(-j, j + 1))
        else:
            top = int(rng.integers(0, max(H - ws, 0) + 1))
            left = int(rng.integers(0, max(W - ws, 0) + 1))
        top = int(np.clip(top, 0, max(H - ws, 0)))
        left = int(np.clip(left, 0, max(W - ws, 0)))
        return top, left, ws

    # ---------------------------------------------------------------- getitem
    def __getitem__(self, idx):
        rec = self.records[idx]
        rng = self._rng(idx)
        H, W = rec["h"], rec["w"]
        top, left, ws = self._window(rec, rng)

        with Image.open(find_image(self.img_dir, rec["id"])) as im:
            im = im.convert("RGB")
            if im.size != (W, H):           # keep RLE geometry authoritative
                im = im.resize((W, H), Image.BILINEAR)
            # crop + resize in a single pass
            pil = im.resize((self.crop, self.crop), Image.BILINEAR,
                            box=(left, top, left + ws, top + ws))

        if self.augment:
            # photometric augmentation only here; the geometric part is applied
            # jointly to image and mask further down so the two stay aligned
            if rng.random() < 0.7:
                pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.8, 1.2)))
            if rng.random() < 0.7:
                pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.8, 1.2)))
            if rng.random() < 0.5:
                pil = ImageEnhance.Color(pil).enhance(float(rng.uniform(0.7, 1.3)))

        img = np.asarray(pil, dtype=np.uint8)
        # gamma augmentation: the worst validation cases are low-light / very
        # flat regions, where the model gives up completely; a random gamma
        # makes the encoder see those areas under many different exposures.
        if self.augment and self.gamma_aug and rng.random() < 0.5:
            g = float(rng.uniform(0.6, 1.6))
            lut = ((np.arange(256) / 255.0) ** g * 255.0).astype(np.uint8)
            img = lut[img]
        mask = rle.decode_window(rec["rle"], H, W, top, top + ws, left, left + ws)
        if mask.shape != (self.crop, self.crop):
            mask = np.asarray(Image.fromarray(mask * 255).resize(
                (self.crop, self.crop), Image.NEAREST), dtype=np.uint8) > 127
        else:
            mask = mask.astype(bool)

        # geometric augmentation applied identically to image and mask
        if self.augment:
            if rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); mask = mask[:, ::-1]
            k = int(rng.integers(0, 4))
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
            if self.noise_sigma > 0 and rng.random() < 0.3:
                noise = rng.normal(0.0, self.noise_sigma, img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        x = _to_tensor(img.transpose(2, 0, 1))
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return {"image": x, "mask": y, "id": rec["id"],
                "orig_hw": torch.tensor([H, W])}


class AIGCFullDataset(Dataset):
    """Full-image dataset for tiled evaluation (uint8 to keep RAM small)."""

    def __init__(self, records, img_dir: str):
        self.records = records
        self.img_dir = img_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        H, W = rec["h"], rec["w"]
        with Image.open(find_image(self.img_dir, rec["id"])) as im:
            im = im.convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.BILINEAR)
            img = np.asarray(im, dtype=np.uint8)
        mask = rle.decode_full(rec["rle"], H, W)
        return {"image": torch.from_numpy(img.transpose(2, 0, 1).copy()),
                "mask": torch.from_numpy(mask), "id": rec["id"]}


def collate_full(batch):
    """batch_size is 1 for the full-image loader."""
    return batch[0]
