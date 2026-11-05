"""Stage 3: confidence intervals that respect the clip structure.

Frames of one clip are near-duplicates, so a bootstrap over frames is too optimistic.
Here whole clips are resampled (within each class, so every resample keeps all three
gestures): the CI of the seed-mean accuracy on the pooled new environments and on the
A1 test set (3 clips — shown only to make the point that 3 clips cannot give a tight CI).

Usage:  python scripts/stage3_clip_ci.py
Writes: results/a2/stage3_clip_ci.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import DATA_V2_DIR, RESULTS_DIR

ORDER = ["B0", "S1", "S2", "S3", "S4-probe", "S4", "S5"]


def main(n_boot=5000, seed=0):
    pred = pd.read_csv(RESULTS_DIR / "stage3_predictions.csv")
    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")[["filepath", "video"]]
    pred = pred.merge(meta, on="filepath")
    pred = pred[pred.test_set != "hagrid_people"]
    pred["group"] = np.where(pred.test_set == "a1_test", "a1_test", "new_envs")
    pred["correct"] = pred.label == pred.pred
    rng = np.random.default_rng(seed)
    rows = []
    for (grp, model), g in pred.groupby(["group", "model"]):
        clip = g.groupby(["video", "label"]).correct.agg(["sum", "size"]).reset_index()  # over seeds and frames
        by_cls = [c for _, c in clip.groupby("label")]
        accs = []
        for _ in range(n_boot):
            res = pd.concat([c.iloc[rng.integers(0, len(c), len(c))] for c in by_cls])
            accs.append(res["sum"].sum() / res["size"].sum())
        rows.append({"test_group": grp, "model": model, "n_clips": len(clip),
                     "acc_seed_mean": clip["sum"].sum() / clip["size"].sum(),
                     "clip_ci_low": np.quantile(accs, 0.025), "clip_ci_high": np.quantile(accs, 0.975)})
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out.model, ORDER)
    out = out.sort_values(["test_group", "model"])
    out.round(4).to_csv(RESULTS_DIR / "stage3_clip_ci.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
