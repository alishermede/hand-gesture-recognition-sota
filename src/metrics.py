"""Metrics: accuracy, macro-F1, per-class report, confusion matrix, bootstrap CIs."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import CLASSES


def summarize(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "n": len(y_true),
        "n_errors": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
    }
    for c in CLASSES:
        out[f"{c}_precision"] = rep[c]["precision"]
        out[f"{c}_recall"] = rep[c]["recall"]
        out[f"{c}_f1"] = rep[c]["f1-score"]
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))


def error_pattern(y_true, y_pred):
    """'thumb->peace:11' style summary of the off-diagonal cells."""
    cm = confusion(y_true, y_pred)
    parts = [f"{CLASSES[i]}->{CLASSES[j]}:{cm[i, j]}"
             for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j and cm[i, j]]
    return ", ".join(parts) or "none"


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI for accuracy, resampling images. Frames within a clip are correlated,
    so this interval is optimistic; the report says so wherever it is quoted."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    accs = (y_true[idx] == y_pred[idx]).mean(1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))
