"""Report figure: the A2 pipeline diagram (Neurohive-style SOTA page).

Usage:  python scripts/make_diagram.py
Writes: figures/a2/pipeline_diagram.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import FIG_DIR, ensure_dirs

BOX = dict(boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.2)
COL = {"input": "#e8eef7", "sota": "#fde9d9", "ours": "#e6f4ea", "train": "#f3e8fd", "out": "#fff7d6"}


def box(ax, x, y, w, h, text, kind, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=COL[kind], edgecolor="#555", **BOX))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), text, ha="center", va="center", fontsize=9.5, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.035, sub, ha="center", va="center", fontsize=7.5, color="#333")


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, lw=1.2,
                                 color="#444", linestyle=ls))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, text, ha="center", fontsize=7.5, color="#444")


def main():
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.01, 0.62, 0.12, 0.16, "Frame", "input", "webcam / photo")
    box(ax, 0.18, 0.62, 0.17, 0.16, "MediaPipe Hand\nLandmarker", "sota", "21 landmarks, ~25 ms CPU")
    box(ax, 0.41, 0.62, 0.15, 0.16, "Square hand crop", "ours", "25% margin;\ncentre crop if no hand")
    box(ax, 0.62, 0.62, 0.17, 0.16, "Classifier", "ours", "ResNet-18 (S1-S3)\nDINOv2 ViT-S/14 (S4)")
    box(ax, 0.85, 0.62, 0.14, 0.16, "fist / peace /\nthumb", "out", "+ Grad-CAM / rollout")

    box(ax, 0.41, 0.30, 0.15, 0.16, "SAM 2.1", "sota", "landmarks = point\nprompts -> hand mask")
    box(ax, 0.62, 0.30, 0.17, 0.16, "Background\nreplacement", "train", "DTD texture / gradient\np = 0.5, train only")
    box(ax, 0.18, 0.06, 0.17, 0.14, "Landmark MLP (S5)", "ours", "63 normalised coords")
    box(ax, 0.41, 0.06, 0.15, 0.14, "HaGRID", "train", "train only, split\nby person (S3+)")

    arrow(ax, 0.13, 0.70, 0.18, 0.70)
    arrow(ax, 0.35, 0.70, 0.41, 0.70, "box")
    arrow(ax, 0.56, 0.70, 0.62, 0.70)
    arrow(ax, 0.79, 0.70, 0.85, 0.70)
    arrow(ax, 0.30, 0.62, 0.45, 0.46, "landmark box", ls="--")
    arrow(ax, 0.485, 0.62, 0.485, 0.46, ls="--")
    arrow(ax, 0.56, 0.38, 0.62, 0.38, "mask", ls="--")
    arrow(ax, 0.705, 0.46, 0.705, 0.62, "augmented crop", ls="--")
    arrow(ax, 0.265, 0.62, 0.265, 0.20, "landmarks", ls="--")
    arrow(ax, 0.56, 0.13, 0.66, 0.30, "extra crops", ls="--")

    for i, (k, lab) in enumerate([("sota", "pretrained SOTA component"), ("ours", "trained / built here"),
                                  ("train", "training-time only"), ("out", "output")]):
        ax.add_patch(plt.Rectangle((0.83, 0.23 - i * 0.055), 0.02, 0.035, facecolor=COL[k], edgecolor="#555"))
        ax.text(0.86, 0.2475 - i * 0.055, lab, va="center", fontsize=8)
    ax.text(0.01, 0.96, "A2 pipeline: solid = inference path, dashed = training-time / alternative paths",
            fontsize=10, weight="bold")
    fig.savefig(FIG_DIR / "pipeline_diagram.png", dpi=150, bbox_inches="tight")
    print("saved", FIG_DIR / "pipeline_diagram.png")


if __name__ == "__main__":
    main()
