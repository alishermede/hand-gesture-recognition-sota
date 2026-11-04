"""Stage 0: reproduce the A1 ResNet-18 result with the refactored src/ code.

Seed 42 is the A1 seed. In the A1 environment (torch 2.14.0+cpu) it should give the
A1 numbers: acc 0.945, macro-F1 0.947, 11 errors, all thumb->peace. Seeds 43 and 44
run the unchanged A1 recipe again to show how much of 0.945 is seed luck.

Usage:
    python scripts/stage0_reproduce.py                    # seeds 42 43 44, auto device
    python scripts/stage0_reproduce.py --seeds 42 --device cpu
    FAST_MODE=1 python scripts/stage0_reproduce.py        # 2+2 epochs smoke test

Writes (to results/a2/, *_fast.csv in FAST mode):
    stage0_reproduce.csv     one row per seed
    stage0_predictions.csv   per image, per seed: true, pred, class probabilities
    stage0_history.csv       training curves
    models/b0_seed{N}.pt     weights, re-used for evaluating B0 on the new test sets
"""

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.config import A1_TEST_CLIPS, CLASSES, FAST, MODELS_DIR, RESULTS_DIR, SEEDS, ensure_dirs, out_name
from src.data import a1_eval_tf, a1_train_tf, load_a1_meta, split_frames
from src.metrics import error_pattern, summarize
from src.models import run_resnet18
from src.train import Timer, predict_proba

A1_REFERENCE = {"acc": 0.945, "macro_f1": 0.9474016769984801, "n_errors": 11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    args = ap.parse_args()

    ensure_dirs()
    device = torch.device(args.device) if args.device else None
    dev_name = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"torch {torch.__version__} | device {dev_name} | fast {FAST} | python {platform.python_version()}")

    meta = load_a1_meta()
    train_df, val_df, test_df = split_frames(meta)
    print(f"train {len(train_df)}  val {len(val_df)}  test {len(test_df)}  test clips {sorted(test_df.video.unique())}")
    assert set(test_df.video) == set(A1_TEST_CLIPS)

    rows, preds, hists = [], [], []
    for seed in args.seeds:
        with Timer() as t:
            model, hist, loaders, n_head = run_resnet18(
                train_df, val_df, test_df, a1_train_tf, a1_eval_tf, seed=seed, device=device
            )
        y, p, prob = predict_proba(model, loaders[2], device=device)
        s = summarize(y, p)
        clip = test_df.video.values
        per_clip = {f"acc_{v}": float((p[clip == v] == y[clip == v]).mean()) for v in A1_TEST_CLIPS}
        row = {
            "model": "B0 ResNet-18 (A1 recipe)", "seed": seed, "device": dev_name,
            "torch": torch.__version__, "fast": FAST,
            "best_val_acc": max(hist["val_acc"][n_head:]), "train_seconds": round(t.seconds, 1),
            **s, "error_pattern": error_pattern(y, p), **per_clip,
        }
        if seed == 42:
            row["matches_a1"] = (round(s["acc"], 4) == A1_REFERENCE["acc"]
                                 and s["n_errors"] == A1_REFERENCE["n_errors"])
        rows.append(row)
        print(f"seed {seed}: acc {s['acc']:.4f}  macro-F1 {s['macro_f1']:.4f}  "
              f"errors {s['n_errors']} ({row['error_pattern']})  {t.seconds:.0f}s")

        pr = test_df[["filepath", "video", "label", "lighting", "position_in_take"]].copy()
        pr["seed"], pr["pred"] = seed, [CLASSES[i] for i in p]
        for j, c in enumerate(CLASSES):
            pr[f"p_{c}"] = prob[:, j]
        preds.append(pr)

        h = pd.DataFrame(hist)
        h["epoch"] = range(1, len(h) + 1)
        h["stage"] = ["head"] * n_head + ["full"] * (len(h) - n_head)
        h["seed"] = seed
        hists.append(h)

        if not FAST:
            torch.save(model.state_dict(), MODELS_DIR / f"b0_seed{seed}.pt")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / out_name("stage0_reproduce"), index=False)
    pd.concat(preds).to_csv(RESULTS_DIR / out_name("stage0_predictions"), index=False)
    pd.concat(hists).to_csv(RESULTS_DIR / out_name("stage0_history"), index=False)

    print("\nA1 reference: acc 0.9450  macro-F1 0.9474  errors 11 (thumb->peace:11)")
    cols = ["seed", "acc", "macro_f1", "n_errors", "error_pattern", "best_val_acc", "train_seconds"]
    print(res[cols].to_string(index=False))
    if len(res) > 1:
        print(f"mean ± std over {len(res)} seeds: acc {res.acc.mean():.4f} ± {res.acc.std(ddof=1):.4f}  "
              f"macro-F1 {res.macro_f1.mean():.4f} ± {res.macro_f1.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
