"""Paths and constants shared by every A2 script and the a2_experiments notebook."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset"              # A1 images + metadata.csv (read-only)
DATA_V2_DIR = ROOT / "dataset_v2"        # A2 additions (new clips, HaGRID, metadata_v2.csv)
RESULTS_DIR = ROOT / "results" / "a2"    # every number in REPORT_2.md comes from here
MODELS_DIR = RESULTS_DIR / "models"      # trained weights (git-ignored)
FIG_DIR = ROOT / "figures" / "a2"

CLASSES = ["fist", "peace", "thumb"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Frozen A1 test set: never used for training, tuning, model selection or as backgrounds.
A1_TEST_CLIPS = ("fist2_dark", "peace2", "thumb2")

A1_SEED = 42
SEEDS = (42, 43, 44)

# FAST_MODE=1 shortens every training stage while debugging (same switch as A1).
FAST = os.environ.get("FAST_MODE") == "1"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs():
    for d in (RESULTS_DIR, MODELS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def out_name(stem, ext="csv"):
    """FAST runs write to *_fast files so smoke tests can never be mistaken for results."""
    return f"{stem}{'_fast' if FAST else ''}.{ext}"
