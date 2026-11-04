"""Stage 2 figure: 6 training examples through the hand pipeline.

Columns: original frame | landmarks + crop window | SAM 2.1 mask | hand crop | background replaced.
Also prints how far the A1 close-up errors reach outside the A1 centre square.

Usage:  python scripts/stage2_figure.py
Writes: figures/a2/stage2_pipeline.png, results/a2/stage2_closeups.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_V2_DIR, FIG_DIR, RESULTS_DIR, ROOT, ensure_dirs
from src.crops import load_textures
from src.hands import replace_background
from src.io_utils import imread_unicode, load_rgb

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


def draw(rgb, r):
    out = rgb.copy()
    h, w = out.shape[:2]
    t = max(2, round(min(h, w) / 250))
    pts = np.array([[r[f"x{j}"] * w, r[f"y{j}"] * h] for j in range(21)]).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 120), t)
    for p in pts:
        cv2.circle(out, tuple(p), t + 1, (255, 255, 255), -1)
    x0, y0, s = int(r.box_x0), int(r.box_y0), int(r.box_side)
    cv2.rectangle(out, (x0, y0), (x0 + s, y0 + s), (255, 200, 0), t)
    return out


def main():
    ensure_dirs()
    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")
    hands = pd.read_csv(DATA_V2_DIR / "hands.csv")
    df = meta.merge(hands, on="filepath")
    ok = df[(df.split == "train") & df.detected & (df.mask_area > 0.05)]
    # own frames only (HaGRID crops can show partial faces): one per class and lighting
    own = ok[ok.source == "own"]
    picks = pd.concat([own[own.lighting == l].groupby("label").sample(1, random_state=1) for l in ("normal", "dark")])
    textures, rng = load_textures(), np.random.default_rng(0)

    titles = ["original", "landmarks + crop window", "SAM 2.1 mask", "hand crop (model input)",
              "background replaced (train only)"]
    fig, axes = plt.subplots(len(picks), 5, figsize=(12, 2.5 * len(picks)))
    for i, r in enumerate(picks.itertuples()):
        frame = load_rgb(ROOT / r.frame_path)
        crop = load_rgb(ROOT / r.crop_path)
        mask = imread_unicode(ROOT / r.mask_path, 0) > 127
        shown_mask = crop.copy()
        shown_mask[mask] = (0.45 * shown_mask[mask] + 0.55 * np.array([0, 255, 120])).astype(np.uint8)
        for j, im in enumerate([frame, draw(frame, df.loc[r.Index]), shown_mask, crop,
                                replace_background(crop, mask, textures, rng)]):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9)
        axes[i, 0].set_title(f"{r.label} ({r.lighting})", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage2_pipeline.png", dpi=110, bbox_inches="tight")
    print("saved", FIG_DIR / "stage2_pipeline.png")

    # close-up analysis: the 11 A1 errors shared by every B0 seed
    pr = pd.read_csv(RESULTS_DIR / "stage0_predictions.csv")
    err = pr[pr.label != pr.pred].groupby("filepath").seed.nunique()
    err = set(err[err == 3].index)
    a1t = df[df.test_set == "a1_test"].copy()
    a1t["filepath_a1"] = a1t.filepath.str.replace("dataset/", "", regex=False)
    a1t["b0_error_all_seeds"] = a1t.filepath_a1.isin(err)
    out = (a1t.groupby(["video", "b0_error_all_seeds"])
              .agg(images=("filepath", "size"), detected=("detected", "mean"),
                   mean_frac_landmarks_outside_centre=("frac_outside_centre", "mean"),
                   any_landmark_outside=("frac_outside_centre", lambda s: (s > 0).mean()))
              .reset_index())
    out.to_csv(RESULTS_DIR / "stage2_closeups.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
