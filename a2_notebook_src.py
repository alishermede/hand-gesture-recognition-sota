# %% [markdown]
# # Hand Gesture Recognition — Assignment 2 experiments
#
# **Computer Vision — Assignment 2 (continuation of A1)**
#
# This notebook runs top to bottom. The heavy work lives in `scripts/` and `src/`;
# every number shown here is read from `results/a2/*.csv`, which those scripts write.
#
# * `A2_RETRAIN=1` re-runs each stage's script before reading its results
#   (otherwise the saved results are used, so the notebook runs in about a minute).
# * `FAST_MODE=1` shortens every training stage (smoke test). FAST runs read and
#   write `*_fast.csv`, so they can never be mistaken for real results.

# %%
%matplotlib inline

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

from src.config import FAST, FIG_DIR, RESULTS_DIR, out_name

RETRAIN = os.environ.get("A2_RETRAIN") == "1"


def run_stage(script, *args, python=sys.executable):
    """Run one stage script and show the tail of its log."""
    proc = subprocess.run([python, script, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print("\n".join(proc.stdout.splitlines()[-25:]))
    if proc.returncode:
        print(proc.stderr[-3000:])
        raise RuntimeError(f"{script} failed")


print(f"python {sys.version.split()[0]} | retrain {RETRAIN} | fast {FAST}")

# %% [markdown]
# ## Stage 0 — reproduce the A1 baseline before changing anything
#
# The A1 code was refactored into `src/` (`data.py`, `train.py`, `models.py`,
# `metrics.py`). Before any A2 change, the refactored code must reproduce the A1
# ResNet-18 result: **acc 0.945, macro-F1 0.947, 11 errors, all thumb→peace**.
#
# Seed 42 is the A1 seed and runs on the A1 environment (torch 2.14.0+cpu), so it
# should match exactly. Seeds 43 and 44 re-run the *unchanged* A1 recipe to show how
# much of 0.945 is seed luck. These three runs are also the **B0** baseline for Stage 3.
#
# A bit-exact reproduction needs the A1 environment (torch 2.14.0+cpu, `requirements.txt`):
# set `A1_PYTHON` to that interpreter. In the A2 environment the numbers can differ slightly.

# %%
if RETRAIN:
    run_stage("scripts/stage0_reproduce.py", "--device", "cpu", python=os.environ.get("A1_PYTHON", sys.executable))

res0 = pd.read_csv(RESULTS_DIR / out_name("stage0_reproduce"))
cols = ["seed", "device", "acc", "macro_f1", "n_errors", "error_pattern",
        "acc_fist2_dark", "acc_peace2", "acc_thumb2", "best_val_acc", "train_seconds"]
display(res0[cols].round(4))
print(f"A1 reference          : acc 0.9450   macro-F1 0.9474   11 errors (thumb->peace:11)")
print(f"seed 42 reproduction  : acc {res0.acc[0]:.4f}   macro-F1 {res0.macro_f1[0]:.4f}   "
      f"{res0.n_errors[0]} errors ({res0.error_pattern[0]})")
print(f"B0 over {len(res0)} seeds    : acc {res0.acc.mean():.4f} ± {res0.acc.std(ddof=1):.4f}   "
      f"macro-F1 {res0.macro_f1.mean():.4f} ± {res0.macro_f1.std(ddof=1):.4f}")

# %% [markdown]
# ### B0 training curves (3 seeds)
#
# The dashed line marks the unfreeze point: 6 head-only epochs, then 12 full fine-tune epochs.
# Validation is the A1 one: the last 20% of every training clip. It is drawn from the
# same recordings as the training frames, so it overstates test accuracy (A1 §5).

# %%
hist0 = pd.read_csv(RESULTS_DIR / out_name("stage0_history"))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for seed, h in hist0.groupby("seed"):
    axes[0].plot(h.epoch, h.train_loss, label=f"train s{seed}")
    axes[0].plot(h.epoch, h.val_loss, ls="--", label=f"val s{seed}")
    axes[1].plot(h.epoch, h.val_acc, label=f"val s{seed}")
n_head = int((hist0[hist0.seed == hist0.seed.iloc[0]].stage == "head").sum())
for ax, title in zip(axes, ["loss", "validation accuracy"]):
    ax.axvline(n_head + 0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_title(f"B0 (A1 recipe) — {title}")
    ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / out_name("stage0_curves", "png"), dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### What the A1 centre crop threw away on the misclassified frames
#
# A1 fed the model the central 1072×1072 square of each 1072×1920 portrait frame.
# The figure re-reads every seed-42 error from the original `.MOV` and outlines that
# square in red: everything outside it was never seen by the model.

# %%
if RETRAIN or not (FIG_DIR / "stage0_errors_fullframe.png").exists():
    run_stage("scripts/inspect_closeups.py")
display(Image(filename=str(FIG_DIR / "stage0_errors_fullframe.png")))

# %% [markdown]
# ## Stage 1 — data
#
# * A1 clips re-read from the `.MOV` files (full portrait frames for the hand pipeline;
#   the A1 224 px centre squares stay B0's input). `dataset/` is never rewritten.
# * New A2 clips from `raw_v2/` — **test only**.
# * HaGRID (fist / peace / like→thumb), **training data + a held-out-people test set**,
#   split by `user_id`; faces blurred before any model sees them.

# %%
if RETRAIN:
    if not (RESULTS_DIR.parents[1] / "dataset_v2" / "hagrid" / "metadata_hagrid.csv").exists():
        run_stage("scripts/download_hagrid.py")
        run_stage("scripts/blur_hagrid_faces.py")
    run_stage("scripts/build_dataset_v2.py")
    run_stage("scripts/stage1_report.py")

display(pd.read_csv(RESULTS_DIR / "stage1_dataset_table.csv"))
display(pd.read_csv(RESULTS_DIR / "stage1_shares.csv"))
display(Image(filename=str(FIG_DIR / "stage1_contact_own.png")))

# %% [markdown]
# ## Stage 2 — hand pipeline cache (MediaPipe → crop → SAM 2.1 mask)

# %%
if RETRAIN:
    run_stage("scripts/prepare_textures.py")
    import torch
    run_stage("scripts/precompute_hands.py", "--device", "cuda" if torch.cuda.is_available() else "cpu")
    run_stage("scripts/stage2_figure.py")

display(pd.read_csv(RESULTS_DIR / "stage2_detection.csv").round(3))
display(Image(filename=str(FIG_DIR / "stage2_pipeline.png")))
print("A1 close-up errors vs the A1 centre square (landmarks outside it):")
display(pd.read_csv(RESULTS_DIR / "stage2_closeups.csv").round(3))

# %% [markdown]
# ## Stage 3 — baseline vs SOTA (3 seeds each)
#
# | model | what changes |
# |---|---|
# | B0 | A1 ResNet-18 on the A1 centre square (Stage 0 weights) |
# | S1 | + hand crop (MediaPipe) |
# | S2 | + background replacement (SAM 2.1 masks, p = 0.5, train only) |
# | S3 | + HaGRID training data |
# | S4-probe / S4 | DINOv2 ViT-S/14 on the S3 pipeline: linear probe / light fine-tune |
# | S5 | MLP on the 21 MediaPipe landmarks (no hand → counted as an error) |

# %%
if RETRAIN:
    run_stage("scripts/stage3_train.py")
    run_stage("scripts/stage3_analyze.py")
    run_stage("scripts/stage3_latency.py")

print("Accuracy per test set, mean ± std over 3 seeds")
display(pd.read_csv(RESULTS_DIR / "stage3_per_env.csv"))
summ = pd.read_csv(RESULTS_DIR / "stage3_summary.csv")
display(summ.round(3))

# %%
print("Ablation: change in accuracy / macro-F1 from each step, paired by seed")
display(pd.read_csv(RESULTS_DIR / "stage3_ablation.csv").round(3))
display(Image(filename=str(FIG_DIR / "stage3_curves.png")))
display(Image(filename=str(FIG_DIR / "stage3_confusion.png")))

# %%
print("CPU latency at batch size 1")
display(pd.read_csv(RESULTS_DIR / "stage3_latency.csv")[["kind", "name", "median_ms", "p95_ms", "fps_median"]])
display(pd.read_csv(RESULTS_DIR / "stage3_model_size.csv"))

# %% [markdown]
# ### The A1 close-up errors, before and after

# %%
display(pd.read_csv(RESULTS_DIR / "stage3_closeups.csv"))
display(Image(filename=str(FIG_DIR / "stage3_closeups_before_after.png")))

# %% [markdown]
# ## Stage 4 — explainability: hand or background?
#
# Grad-CAM (ResNet-18) and attention rollout (DINOv2) on each model's own input, and the
# share of the heatmap that falls on the SAM 2.1 hand mask. The mask's area is the
# chance level (a heatmap spread evenly over the image).

# %%
if RETRAIN:
    run_stage("scripts/stage4_explain.py")

display(pd.read_csv(RESULTS_DIR / "stage4_summary.csv").round(3))
display(Image(filename=str(FIG_DIR / "stage4_heatmaps.png")))
