"""Stage 0 follow-up: what did the A1 centre crop throw away on the misclassified frames?

A1 fed the model the central 1072x1072 square of each 1072x1920 portrait frame.
For every B0 seed-42 error this script re-reads the original frame from the .MOV
and draws the square the model actually saw, so we can check whether the fist that
"leaves the frame" in the close-ups is really outside the camera view or only
outside the centre crop.

Writes figures/a2/stage0_errors_fullframe.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIG_DIR, RESULTS_DIR, ROOT, ensure_dirs
from src.io_utils import load_rgb


def read_frame(video, idx):
    cap = cv2.VideoCapture(str(ROOT / f"{video}.MOV"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"could not read frame {idx} of {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    ensure_dirs()
    pr = pd.read_csv(RESULTS_DIR / "stage0_predictions.csv")
    err = pr[(pr.seed == 42) & (pr.label != pr.pred)].reset_index(drop=True)
    print(f"{len(err)} errors for seed 42")
    if err.empty:
        return

    n = len(err)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 6.2), gridspec_kw={"height_ratios": [1, 1.79]})
    axes = axes.reshape(2, n)
    for i, r in err.iterrows():
        idx = int(r.filepath.split("_f")[-1].split(".")[0])
        full = read_frame(r.video, idx)
        h, w = full.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        cv2.rectangle(full, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 40, 40), 12)

        axes[0, i].imshow(load_rgb(ROOT / "dataset" / r.filepath))
        axes[0, i].set_title(f"{r.label}->{r.pred}\np={r[f'p_{r.pred}']:.2f}", fontsize=8)
        axes[1, i].imshow(full)
        axes[1, i].set_title(f"frame {idx}", fontsize=8)
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].text(-0.15, 0.5, "model input\n(A1 centre crop)", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8)
    axes[1, 0].text(-0.15, 0.5, "full frame,\nred = what the\nmodel saw", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("B0 (A1 ResNet-18, seed 42) errors: model input vs. the original portrait frame")
    fig.tight_layout()
    out = FIG_DIR / "stage0_errors_fullframe.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
