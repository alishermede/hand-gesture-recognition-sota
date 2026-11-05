"""Metadata, splits, transforms and the Dataset class.

The A1 protocol is reproduced exactly:
  * test  = the three frozen held-out clips (A1_TEST_CLIPS)
  * val   = the last 20% (in time) of every remaining clip
  * train = the first 80% of every remaining clip
"""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import A1_TEST_CLIPS, CLASS_TO_IDX, DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
from .io_utils import load_rgb

VAL_TAIL = 0.20


def load_a1_meta():
    """dataset/metadata.csv with the A1 train/val/test split and an absolute `path` column."""
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    meta["path"] = [str(DATA_DIR / p) for p in meta.filepath]
    meta["split"] = [
        "test" if v in A1_TEST_CLIPS else ("val" if pos >= 1 - VAL_TAIL else "train")
        for v, pos in zip(meta.video, meta.position_in_take)
    ]
    meta["source"] = "own"
    meta["environment"] = "a1_indoor"
    return meta


def split_frames(meta):
    return tuple(meta[meta.split == s].reset_index(drop=True) for s in ("train", "val", "test"))


# --- transforms (identical to A1) -------------------------------------------------

a1_train_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ]
)

a1_eval_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def denorm(t):
    x = t.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(x, 0, 1)


class GestureDataset(Dataset):
    """Rows of a metadata frame -> (tensor, class index). `df.path` holds absolute paths."""

    def __init__(self, df, tf):
        self.paths = list(df.path)
        self.targets = np.array([CLASS_TO_IDX[c] for c in df.label])
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_rgb(self.paths[i])), int(self.targets[i])


def make_loaders(train_df, val_df, test_df, train_tf, eval_tf, bs=32):
    # num_workers=0: Windows + notebook multiprocessing is not worth the trouble here.
    return (
        DataLoader(GestureDataset(train_df, train_tf), batch_size=bs, shuffle=True, num_workers=0),
        DataLoader(GestureDataset(val_df, eval_tf), batch_size=bs, num_workers=0),
        DataLoader(GestureDataset(test_df, eval_tf), batch_size=bs, num_workers=0),
    )
