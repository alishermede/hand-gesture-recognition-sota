"""Stage 3 analysis: tables and figures from stage3_runs.csv / stage3_predictions.csv.

  stage3_summary.csv    model x test set: acc / macro-F1 mean ± std over seeds, 95% bootstrap CI
  stage3_per_env.csv    the per-environment table (acc mean ± std), models as rows
  stage3_ablation.csv   Δ acc / Δ macro-F1 from crop, background replacement, HaGRID, backbone,
                        paired by seed (same seed, previous model)
  stage3_per_class.csv  per-class precision / recall / F1, mean over seeds
  figures: stage3_confusion.png, stage3_curves.png, stage3_closeups_before_after.png

"new_envs" pools every new-environment test set (all clips recorded for A2).

Usage:  python scripts/stage3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import CLASSES, DATA_V2_DIR, FIG_DIR, RESULTS_DIR, ROOT, ensure_dirs
from src.io_utils import load_rgb
from src.metrics import summarize

ORDER = ["B0", "S1", "S2", "S3", "S4-probe", "S4", "S5"]
FIXED_SETS = ["a1_test", "hagrid_people"]
STEPS = [("crop", "S1", "B0"), ("background replacement", "S2", "S1"), ("HaGRID data", "S3", "S2"),
         ("DINOv2 backbone (vs ResNet-18)", "S4", "S3")]


def with_pooled(pred):
    """Add a 'new_envs' test set pooling every A2 test clip (not A1, not HaGRID)."""
    new = pred[~pred.test_set.isin(FIXED_SETS)]
    if new.empty:
        return pred
    return pd.concat([pred, new.assign(test_set="new_envs")], ignore_index=True)


def per_run(pred):
    rows = []
    for (m, s, t), g in pred.groupby(["model", "seed", "test_set"]):
        y = g.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        p = g.pred.map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).to_numpy()
        rows.append({"model": m, "seed": s, "test_set": t, **summarize(y, p), "detection_rate": g.detected.mean()})
    return pd.DataFrame(rows)


def boot_ci(g, n_boot=2000, seed=0):
    """CI of seed-mean accuracy: per image, the share of seeds that got it right; resample images."""
    c = (g.label == g.pred).groupby(g.filepath).mean().to_numpy()
    rng = np.random.default_rng(seed)
    acc = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1)
    return np.quantile(acc, 0.025), np.quantile(acc, 0.975)


def pm(m, s):
    return f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}"


def main():
    ensure_dirs()
    pred = with_pooled(pd.read_csv(RESULTS_DIR / "stage3_predictions.csv"))
    runs = per_run(pred)
    sets = [t for t in ["a1_test", "new_envs", "hagrid_people"] if t in set(runs.test_set)] + \
           sorted(set(runs.test_set) - {"a1_test", "new_envs", "hagrid_people"})
    models = [m for m in ORDER if m in set(runs.model)]

    summ = []
    for (m, t), g in runs.groupby(["model", "test_set"]):
        lo, hi = boot_ci(pred[(pred.model == m) & (pred.test_set == t)])
        summ.append({"model": m, "test_set": t, "n_images": int(g.n.iloc[0]), "n_seeds": len(g),
                     "acc_mean": g.acc.mean(), "acc_std": g.acc.std(ddof=1), "acc_ci_low": lo, "acc_ci_high": hi,
                     "f1_mean": g.macro_f1.mean(), "f1_std": g.macro_f1.std(ddof=1),
                     "detection_rate": g.detection_rate.mean()})
    summ = pd.DataFrame(summ)
    summ["model"] = pd.Categorical(summ.model, models)
    summ = summ.sort_values(["test_set", "model"])
    summ.round(4).to_csv(RESULTS_DIR / "stage3_summary.csv", index=False)

    env = summ.assign(cell=[pm(a, b) for a, b in zip(summ.acc_mean, summ.acc_std)]) \
              .pivot(index="model", columns="test_set", values="cell").reindex(index=models, columns=sets)
    env.to_csv(RESULTS_DIR / "stage3_per_env.csv")
    print("accuracy, mean ± std over seeds\n", env.to_string())

    abl = []
    for step, new, old in STEPS:
        for t in sets:
            a = runs[(runs.model == new) & (runs.test_set == t)].set_index("seed")
            b = runs[(runs.model == old) & (runs.test_set == t)].set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            d_acc, d_f1 = a.loc[common, "acc"] - b.loc[common, "acc"], a.loc[common, "macro_f1"] - b.loc[common, "macro_f1"]
            abl.append({"step": step, "model": f"{new} vs {old}", "test_set": t, "n_seeds": len(common),
                        "d_acc_mean": d_acc.mean(), "d_acc_std": d_acc.std(ddof=1),
                        "d_f1_mean": d_f1.mean(), "d_f1_std": d_f1.std(ddof=1)})
    abl = pd.DataFrame(abl)
    abl.round(4).to_csv(RESULTS_DIR / "stage3_ablation.csv", index=False)
    print("\nablation (paired by seed)\n", abl.round(3).to_string(index=False))

    pc = runs.groupby(["model", "test_set"])[[f"{c}_{k}" for c in CLASSES for k in ("precision", "recall", "f1")]].mean()
    pc.round(4).to_csv(RESULTS_DIR / "stage3_per_class.csv")

    # confusion matrices, summed over seeds
    show = [m for m in ["B0", "S3", "S4", "S5"] if m in models]
    cm_sets = [t for t in ["a1_test", "new_envs", "hagrid_people"] if t in sets]
    fig, axes = plt.subplots(len(cm_sets), len(show), figsize=(3.3 * len(show), 3.1 * len(cm_sets)), squeeze=False)
    for i, t in enumerate(cm_sets):
        for j, m in enumerate(show):
            g = pred[(pred.model == m) & (pred.test_set == t)]
            labels = CLASSES + (["none"] if (g.pred == "none").any() else [])
            cm = confusion_matrix(g.label, g.pred, labels=labels)[: len(CLASSES)]
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues")
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, v, ha="center", va="center", fontsize=8, color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks(range(len(labels)), labels, fontsize=7)
            ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
            ax.set_title(f"{m} · {t}", fontsize=9)
            if j == 0:
                ax.set_ylabel("true")
            ax.set_xlabel("predicted (summed over seeds)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # training curves: validation accuracy and loss, mean over seeds
    hist = pd.read_csv(RESULTS_DIR / "stage3_history.csv")
    h0 = pd.read_csv(RESULTS_DIR / "stage0_history.csv").assign(model="B0")
    hist = pd.concat([h0, hist], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for m in [x for x in ["B0", "S1", "S2", "S3", "S4", "S5"] if x in set(hist.model)]:
        h = hist[hist.model == m].groupby("epoch")[["train_loss", "val_loss", "val_acc"]].mean()
        axes[0].plot(h.index, h.train_loss, label=m)
        axes[1].plot(h.index, h.val_loss, label=m)
        axes[2].plot(h.index, h.val_acc, label=m)
    for ax, t in zip(axes, ["train loss", "validation loss", "validation accuracy"]):
        ax.set_title(f"{t} (mean over seeds)")
        ax.set_xlabel("epoch")
        ax.axvline(6.5, color="grey", ls=":", lw=1)
        ax.legend(fontsize=7)
    axes[0].set_xlim(0, 20)
    axes[1].set_xlim(0, 20)
    axes[2].set_xlim(0, 20)
    fig.suptitle("Image models: 6 head-only epochs, then fine-tuning (dotted line). S5 (MLP) trains 300 epochs; first 20 shown.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    closeups(pred, summ)


def closeups(pred, summ):
    """Before/after on the 11 A1 frames that B0 gets wrong with every seed."""
    p0 = pd.read_csv(RESULTS_DIR / "stage0_predictions.csv")
    e = p0[p0.label != p0.pred].groupby("filepath").seed.nunique()
    err = sorted("dataset/" + e[e == 3].index)
    a1 = summ[(summ.test_set == "a1_test") & (summ.model.isin(["S3", "S4"]))]
    best = str(a1.sort_values("acc_mean").model.iloc[-1]) if len(a1) else "S3"
    hands = pd.read_csv(DATA_V2_DIR / "hands.csv").set_index("filepath")

    rows = []
    for f in err:
        b0 = pred[(pred.model == "B0") & (pred.filepath == f)]
        bm = pred[(pred.model == best) & (pred.filepath == f)]
        rows.append({"filepath": f, "b0_correct_seeds": int((b0.pred == b0.label).sum()),
                     f"{best}_correct_seeds": int((bm.pred == bm.label).sum()), "n_seeds": len(bm),
                     "detected": bool(hands.loc[f, "detected"])})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stage3_closeups.csv", index=False)
    print(f"\nA1 close-up errors (B0 wrong with all 3 seeds): {best} correct in "
          f"{out[f'{best}_correct_seeds'].sum()} of {out.n_seeds.sum()} (image, seed) cases")

    n = len(out)
    fig, axes = plt.subplots(2, n, figsize=(1.75 * n, 4.4), squeeze=False)
    for i, r in out.iterrows():
        axes[0, i].imshow(load_rgb(ROOT / r.filepath))
        axes[0, i].set_title(f"B0 {r.b0_correct_seeds}/3", fontsize=8, color="red")
        axes[1, i].imshow(load_rgb(ROOT / hands.loc[r.filepath, "crop_path"]))
        ok = r[f"{best}_correct_seeds"]
        axes[1, i].set_title(f"{best} {ok}/{r.n_seeds}" + ("" if r.detected else "\n(no hand: centre)"),
                             fontsize=8, color="green" if ok == r.n_seeds else ("orange" if ok else "red"))
        for ax in axes[:, i]:
            ax.axis("off")
    fig.suptitle(f"A1 close-up errors (true: thumb). Top: B0 input. Bottom: {best} hand crop. "
                 "Title = seeds that classify it correctly.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_closeups_before_after.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
