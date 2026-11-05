"""Stage 4: explainability. Is the model looking at the hand or at the background?

For every test image with a detected hand (A1 test + new environments), seed 42 models:
  * B0  Grad-CAM on its input (the A1 centre square)
  * S2  Grad-CAM on its input (the hand crop) — same images as B0
  * S3  Grad-CAM, S4 attention rollout (DINOv2) on the hand crop
and a SAM 2.1 hand mask computed *for each model's own input*. The score is the share
of the heatmap mass on the hand; the mask's area is the chance level.

Usage:  python scripts/stage4_explain.py [--device cuda]
Writes: results/a2/stage4_heat_on_hand.csv (per image), stage4_summary.csv,
        figures/a2/stage4_heatmaps.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import CLASSES, FIG_DIR, MODELS_DIR, RESULTS_DIR, ROOT, ensure_dirs
from src.crops import load_v2
from src.data import a1_eval_tf
from src.explain import attention_rollout, energy_on_mask, gradcam, overlay
from src.hands import HandSegmenter, centre_box, landmarks_in_crop
from src.io_utils import load_rgb
from src.models import DinoV2Classifier, resnet18_classifier


def landmarks(r):
    return np.array([[r[f"x{j}"], r[f"y{j}"], r[f"z{j}"]] for j in range(21)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ensure_dirs()
    dev = torch.device(args.device)

    df = load_v2()
    df = df[(df.split == "test") & (df.source == "own") & df.detected].reset_index(drop=True)
    print(f"{len(df)} own test images with a detected hand, sets: {sorted(df.test_set.unique())}")

    def load(name, cls):
        m = cls()
        m.load_state_dict(torch.load(MODELS_DIR / f"{name}_seed{args.seed}.pt", map_location="cpu"))
        return m.to(dev).eval()

    models = {"B0": load("b0", resnet18_classifier), "S2": load("s2", resnet18_classifier),
              "S3": load("s3", resnet18_classifier),
              "S4": load("s4", lambda: DinoV2Classifier(attn_implementation="eager"))}
    seg = HandSegmenter(args.device)

    show_idx = set(df.groupby("label").sample(2, random_state=0).index)  # figure examples
    rows, keep = [], {}
    for i, r in df.iterrows():
        frame = load_rgb(ROOT / r.frame_path)
        h, w = frame.shape[:2]
        lm = landmarks(r)
        sq_box = centre_box(w, h)
        x0, y0, s = sq_box
        inputs = {
            "B0": (frame[y0:y0 + s, x0:x0 + s], sq_box),
            "crop": (load_rgb(ROOT / r.crop_path), (r.box_x0, r.box_y0, r.box_side)),
        }
        masks = {}
        for k, (img, box) in inputs.items():
            size = img.shape[0]
            masks[k], _ = seg.segment(np.ascontiguousarray(img), landmarks_in_crop(lm, w, h, box, size))

        for name, m in models.items():
            img, mk = (inputs["B0"][0], masks["B0"]) if name == "B0" else (inputs["crop"][0], masks["crop"])
            x = a1_eval_tf(img).unsqueeze(0).to(dev)
            if name == "S4":
                heat = attention_rollout(m, x)
                with torch.no_grad():
                    pred = int(m(x).argmax(1))
            else:
                heat, pred = gradcam(m, x)
            on, area = energy_on_mask(heat, mk)
            rows.append({"filepath": r.filepath, "test_set": r.test_set, "label": r.label, "model": name,
                         "method": "attention rollout" if name == "S4" else "Grad-CAM",
                         "pred": CLASSES[pred], "heat_on_hand": on, "hand_area": area})
            if i in show_idx:
                keep.setdefault(i, {})[name] = (img, heat, CLASSES[pred], on)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}")

    per = pd.DataFrame(rows)
    per.to_csv(RESULTS_DIR / "stage4_heat_on_hand.csv", index=False)
    per["lift"] = per.heat_on_hand / per.hand_area
    summ = per.groupby(["model", "method"]).agg(images=("filepath", "size"), heat_on_hand=("heat_on_hand", "mean"),
                                                 hand_area=("hand_area", "mean"), lift=("lift", "mean"),
                                                 acc=("pred", lambda p: (p == per.loc[p.index, "label"]).mean())).reset_index()
    summ.round(4).to_csv(RESULTS_DIR / "stage4_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(per.groupby(["test_set", "model"]).heat_on_hand.mean().unstack().round(3).to_string())

    # figure: 2 images per class from different clips, B0 vs S2 (+ S4)
    picks = sorted(keep, key=lambda k: CLASSES.index(df.loc[k, "label"]))
    cols = ["B0", "S2", "S4"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.0 * len(cols), 3.1 * len(picks)), squeeze=False)
    for i, idx in enumerate(picks):
        for j, name in enumerate(cols):
            img, heat, pred, on = keep[idx][name]
            ax = axes[i, j]
            ax.imshow(overlay(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), heat))
            ok = pred == df.loc[idx, "label"]
            ax.set_title(f"{name}: pred {pred} · {on:.0%} on hand", fontsize=8, color="green" if ok else "red")
            ax.axis("off")
        axes[i, 0].text(-12, 112, f"true: {df.loc[idx, 'label']}\n{df.loc[idx, 'test_set']}", rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle("Grad-CAM (B0, S2) and attention rollout (S4) on each model's own input, seed 42", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage4_heatmaps.png", dpi=110, bbox_inches="tight")
    print("saved", FIG_DIR / "stage4_heatmaps.png")


if __name__ == "__main__":
    main()
