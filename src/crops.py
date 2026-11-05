"""Stage 3 data: dataset_v2 rows joined with the Stage 2 hand cache, and the Dataset classes.

  * SquareDataset  — the A1 input (centre square): used by B0
  * CropDataset    — the cached 256x256 hand crop, with optional background replacement
                     (train only, p = bg_p) from the DTD texture bank
  * landmark_features — normalised 21-landmark vectors for the landmark MLP (S5)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CLASS_TO_IDX, DATA_V2_DIR, ROOT
from .hands import replace_background
from .io_utils import imread_unicode, load_rgb


def load_v2(fast_hands=False):
    """metadata_v2 + hands cache, with absolute paths. One row per image."""
    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")
    hands = pd.read_csv(DATA_V2_DIR / ("hands_fast.csv" if fast_hands else "hands.csv"))
    df = meta.merge(hands, on="filepath", how="inner", validate="one_to_one")
    df["path"] = [str(ROOT / p) for p in df.filepath]             # B0 input
    df["crop_abs"] = [str(ROOT / p) for p in df.crop_path]        # S1+ input
    df["mask_abs"] = [str(ROOT / p) if isinstance(p, str) and p else "" for p in df.mask_path.fillna("")]
    df["test_set"] = df.test_set.fillna("")
    return df


def splits(df, use_hagrid):
    train = df[(df.split == "train") & (use_hagrid | (df.source == "own"))].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    tests = {name: g.reset_index(drop=True) for name, g in df[df.split == "test"].groupby("test_set")}
    return train, val, tests


def load_textures(folder=ROOT / "data_ext" / "textures"):
    return [load_rgb(p) for p in sorted(folder.glob("*.jpg"))]


class SquareDataset(Dataset):
    def __init__(self, df, tf):
        self.paths, self.tf = list(df.path), tf
        self.targets = np.array([CLASS_TO_IDX[c] for c in df.label])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_rgb(self.paths[i])), int(self.targets[i])


class CropDataset(Dataset):
    def __init__(self, df, tf, bg_p=0.0, textures=None, seed=0):
        self.crops, self.masks = list(df.crop_abs), list(df.mask_abs)
        self.targets = np.array([CLASS_TO_IDX[c] for c in df.label])
        self.tf, self.bg_p, self.textures = tf, bg_p, textures or []
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        img = load_rgb(self.crops[i])
        if self.bg_p and self.masks[i] and self.rng.random() < self.bg_p:
            mask = imread_unicode(self.masks[i], 0) > 127
            if mask.mean() > 0.03:  # skip degenerate masks
                img = replace_background(img, mask, self.textures, self.rng)
        return self.tf(img), int(self.targets[i])


# --- landmarks ------------------------------------------------------------------------

def landmark_array(df):
    """(n, 21, 3) landmarks in pixel units of the frame (x, y) and the same scale for z."""
    x = df[[f"x{j}" for j in range(21)]].to_numpy(np.float32) * df.frame_w.to_numpy()[:, None]
    y = df[[f"y{j}" for j in range(21)]].to_numpy(np.float32) * df.frame_h.to_numpy()[:, None]
    z = df[[f"z{j}" for j in range(21)]].to_numpy(np.float32) * df.frame_w.to_numpy()[:, None]
    return np.stack([x, y, z], -1)


def normalise_landmarks(lm):
    """Translate to the wrist and scale by the largest wrist distance: position/size invariant."""
    lm = lm - lm[:, :1]
    scale = np.linalg.norm(lm[:, :, :2], axis=-1).max(1, keepdims=True)[..., None]
    return lm / np.maximum(scale, 1e-6)


def augment_landmarks(lm, rng):
    """Random in-plane rotation (+-20 deg), scale, horizontal flip and jitter, train only."""
    n = len(lm)
    a = rng.uniform(-np.pi / 9, np.pi / 9, n)
    rot = np.stack([np.stack([np.cos(a), -np.sin(a)], -1), np.stack([np.sin(a), np.cos(a)], -1)], -2)
    xy = np.einsum("nij,nkj->nki", rot, lm[:, :, :2]) * rng.uniform(0.9, 1.1, (n, 1, 1))
    flip = rng.random(n) < 0.5
    xy[flip, :, 0] *= -1
    out = np.concatenate([xy, lm[:, :, 2:]], -1) + rng.normal(0, 0.01, lm.shape)
    return out.astype(np.float32)


class LandmarkMLP(torch.nn.Module):
    def __init__(self, n_classes=3, d_in=63):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))
