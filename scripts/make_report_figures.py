"""Report figures: comparison charts built only from results/a2/*.csv.

  report_env_accuracy.png     accuracy per test set, every model (mean ± std over 3 seeds)
  report_ablation.png         B0 -> S4 step by step on the new environments (waterfall)
  report_accuracy_latency.png accuracy on new environments vs CPU end-to-end latency
  report_heat_on_hand.png     share of the heatmap on the hand, per environment

Usage:  python scripts/make_report_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIG_DIR, RESULTS_DIR, ensure_dirs

# reference palette (dataviz skill, light surface); validated: blue / orange / aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BASE = "#898781"          # muted ink: the A1 baseline
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9,
})

MODELS = ["B0", "S1", "S2", "S3", "S4", "S5"]
NAMES = {"B0": "B0  A1 baseline", "S1": "S1  + hand crop", "S2": "S2  + bg replacement",
         "S3": "S3  + HaGRID", "S4": "S4  DINOv2", "S5": "S5  landmark MLP"}
COLOR = {"B0": BASE, "S1": BLUE, "S2": BLUE, "S3": BLUE, "S4": ORANGE, "S5": AQUA}
SETS = [("a1_test", "A1 test (frozen)\n3 clips"), ("curtains", "curtains\n6 clips"),
        ("night", "night window\n6 clips"), ("table", "cluttered table\n6 clips"),
        ("hagrid_people", "HaGRID unseen people\n292 people")]


def env_accuracy(summ):
    fig, axes = plt.subplots(1, len(SETS), figsize=(14, 3.6), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for ax, (t, title) in zip(axes, SETS):
        s = summ[summ.test_set == t].set_index("model")
        acc = [s.loc[m, "acc_mean"] for m in MODELS]
        sd = [s.loc[m, "acc_std"] for m in MODELS]
        ax.barh(y, acc, height=0.68, color=[COLOR[m] for m in MODELS], xerr=sd,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for yi, a, e in zip(y, acc, sd):
            ax.text(a + e + 0.03, yi, f"{a:.2f}", va="center", fontsize=8, color=INK)
        ax.axvline(1 / 3, color=AXIS, lw=0.8, ls="--")
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [NAMES[m] for m in MODELS])
    axes[0].text(1 / 3, -0.9, "chance", fontsize=7, color=INK2, ha="center")
    fig.suptitle("Accuracy per test set (mean ± std over 3 seeds). All new-environment clips are test-only.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_env_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation(summ, abl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, t, title in [(axes[0], "new_envs", "New environments (18 clips, 1,080 frames)"),
                         (axes[1], "hagrid_people", "HaGRID unseen people (292)")]:
        start = summ[(summ.test_set == t) & (summ.model == "B0")].acc_mean.iloc[0]
        steps = abl[abl.test_set == t].set_index("step")
        labels = ["B0", "+ hand\ncrop", "+ bg\nreplacement", "+ HaGRID\ndata", "ResNet →\nDINOv2", "S4"]
        deltas = [steps.loc[s, "d_acc_mean"] for s in
                  ["crop", "background replacement", "HaGRID data", "DINOv2 backbone (vs ResNet-18)"]]
        stds = [steps.loc[s, "d_acc_std"] for s in
                ["crop", "background replacement", "HaGRID data", "DINOv2 backbone (vs ResNet-18)"]]
        level = start
        ax.bar(0, start, color=BASE, width=0.6)
        ax.text(0, start + 0.015, f"{start:.3f}", ha="center", fontsize=8)
        for i, (d, sd) in enumerate(zip(deltas, stds), start=1):
            bottom = level if d >= 0 else level + d
            ax.bar(i, abs(d), bottom=bottom, color=BLUE if d >= 0 else ORANGE, width=0.6)
            ax.text(i, max(level, level + d) + 0.015, f"{d:+.3f}\n±{sd:.3f}", ha="center", fontsize=7.5)
            level += d
        ax.bar(5, level, color="#1c5cab", width=0.6)
        ax.text(5, level + 0.015, f"{level:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(6), labels, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Ablation: what each step adds (mean Δ accuracy, paired by seed; blue = gain, orange = loss)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def acc_latency(summ, lat):
    rows = {"B0": "B0: centre square + ResNet-18", "S3": "S3: MediaPipe + crop + ResNet-18",
            "S4": "S4: MediaPipe + crop + DINOv2", "S5": "S5: MediaPipe + landmark MLP"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m, name in rows.items():
        x = lat[lat.name == name].median_ms.iloc[0]
        r = summ[(summ.test_set == "new_envs") & (summ.model == m)].iloc[0]
        ax.errorbar(x, r.acc_mean, yerr=r.acc_std, fmt="o", ms=8, color=COLOR[m], ecolor=INK2, elinewidth=0.8,
                    capsize=2, markeredgecolor=SURF, markeredgewidth=2)
        ax.annotate(f"{m}  {r.acc_mean:.3f} @ {x:.0f} ms", (x, r.acc_mean), xytext=(8, -3 if m != "S5" else 6),
                    textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 120)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("CPU end-to-end latency, median ms (batch 1, Ryzen 5 5600H)")
    ax.set_ylabel("accuracy, new environments")
    ax.set_title("Accuracy vs latency (SAM 2.1 adds ~2.2 s and is optional)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_accuracy_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def heat(per):
    sets = ["a1_test", "curtains", "night", "table"]
    models = [("B0", BASE), ("S2", BLUE), ("S4", ORANGE)]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.26
    for i, (m, c) in enumerate(models):
        vals = [per[(per.model == m) & (per.test_set == s)].heat_on_hand.mean() for s in sets]
        xs = np.arange(len(sets)) + (i - 1) * w
        ax.bar(xs, vals, width=w - 0.03, color=c, label={"B0": "B0 Grad-CAM", "S2": "S2 Grad-CAM",
                                                         "S4": "S4 attention rollout"}[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    area = [per[(per.model == "S2") & (per.test_set == s)].hand_area.mean() for s in sets]
    for j, a in enumerate(area):
        ax.plot([j - 0.42, j + 0.42], [a, a], color=INK, lw=1, ls=":")
    ax.plot([], [], color=INK, lw=1, ls=":", label="hand area = chance level")
    ax.set_xticks(range(len(sets)), ["A1 test", "curtains", "night window", "cluttered table"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("share of heatmap on the hand")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    ax.set_title("Is the model looking at the hand? (SAM 2.1 mask of each model's own input, seed 42)",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_heat_on_hand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    summ = pd.read_csv(RESULTS_DIR / "stage3_summary.csv")
    abl = pd.read_csv(RESULTS_DIR / "stage3_ablation.csv")
    lat = pd.read_csv(RESULTS_DIR / "stage3_latency.csv")
    per = pd.read_csv(RESULTS_DIR / "stage4_heat_on_hand.csv")
    env_accuracy(summ)
    ablation(summ, abl)
    acc_latency(summ, lat)
    heat(per)
    print("saved 4 report figures to", FIG_DIR)


if __name__ == "__main__":
    main()
