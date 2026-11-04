"""Stage 3: baseline vs SOTA, 3 seeds each, evaluated on every test set.

Models (each change is added on top of the previous one, so the deltas form the ablation):
  B0  A1 ResNet-18 on the A1 centre square (weights from Stage 0, not retrained)
  S1  + hand crop (MediaPipe, 25% margin; centre-square fallback when no hand is found)
  S2  + background replacement with SAM 2.1 masks (p = 0.5, train only)
  S3  + HaGRID training data (person-split, ~400 crops per class)
  S4  DINOv2 ViT-S/14 on the S3 pipeline: linear probe (S4-probe), then light fine-tune
  S5  MLP on the 21 normalised MediaPipe landmarks (own + HaGRID train); an image where
      no hand is detected counts as an error

Test sets: a1_test (frozen), every new environment, a1_room_dark (thumb2_dark),
hagrid_people (held-out HaGRID users).

Usage:  python scripts/stage3_train.py [--models S1 S2 ...] [--seeds 42 43 44] [--device cuda]
        FAST_MODE=1 ...   2+2 epochs, *_fast outputs
Writes: results/a2/stage3_runs.csv, stage3_predictions.csv, stage3_history.csv, models/<m>_seed<N>.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CLASSES, CLASS_TO_IDX, FAST, MODELS_DIR, RESULTS_DIR, SEEDS, ensure_dirs, out_name
from src.crops import (CropDataset, LandmarkMLP, SquareDataset, augment_landmarks, landmark_array,
                       load_textures, load_v2, normalise_landmarks, splits)
from src.data import a1_eval_tf, a1_train_tf
from src.metrics import error_pattern, summarize
from src.models import (DinoV2Classifier, resnet18_classifier, run_two_stage, unfreeze_all,
                        unfreeze_dino_top)
from src.train import Timer, predict_proba, seed_everything

MODEL_DESC = {
    "B0": "ResNet-18, A1 centre square (A1 baseline)",
    "S1": "ResNet-18 + hand crop",
    "S2": "ResNet-18 + crop + background replacement",
    "S3": "ResNet-18 + crop + bg + HaGRID",
    "S4-probe": "DINOv2 ViT-S/14 linear probe (S3 pipeline)",
    "S4": "DINOv2 ViT-S/14 light fine-tune (S3 pipeline)",
    "S5": "Landmark MLP (21 MediaPipe landmarks)",
}


class Recorder:
    """Collects metrics / predictions / curves and rewrites the CSVs after every run."""

    def __init__(self):
        self.runs, self.preds, self.hists = [], [], []

    def evaluate(self, model_name, seed, test_name, df, y, p, prob, extra=None):
        s = summarize(y, p)
        self.runs.append({"model": model_name, "desc": MODEL_DESC[model_name], "seed": seed,
                          "test_set": test_name, **s, "error_pattern": error_pattern(y, p),
                          "detection_rate": float(df.detected.mean()), **(extra or {})})
        pr = df[["filepath", "label", "test_set", "detected"]].copy()
        pr["model"], pr["seed"], pr["pred"] = model_name, seed, [CLASSES[i] if i >= 0 else "none" for i in p]
        for j, c in enumerate(CLASSES):
            pr[f"p_{c}"] = prob[:, j]
        self.preds.append(pr)
        print(f"    {model_name:8s} s{seed} {test_name:14s} n={s['n']:4d} acc {s['acc']:.3f} "
              f"F1 {s['macro_f1']:.3f}  det {df.detected.mean():.0%}")

    def history(self, model_name, seed, hist, n_head):
        h = pd.DataFrame(hist)
        h["epoch"] = range(1, len(h) + 1)
        h["stage"] = ["head"] * n_head + ["full"] * (len(h) - n_head)
        h["model"], h["seed"] = model_name, seed
        self.hists.append(h)

    def save(self):
        merge_save(pd.DataFrame(self.runs), "stage3_runs", ["model", "seed", "test_set"])
        if self.preds:
            merge_save(pd.concat(self.preds), "stage3_predictions", ["model", "seed", "test_set", "filepath"])
        if self.hists:
            merge_save(pd.concat(self.hists), "stage3_history", ["model", "seed", "epoch"])


def merge_save(new, stem, keys):
    """Replace rows of the re-run (model, seed) pairs, keep everything else already on disk."""
    path = RESULTS_DIR / out_name(stem)
    if path.exists():
        old = pd.read_csv(path)
        done = new[["model", "seed"]].drop_duplicates()
        old = old.merge(done, on=["model", "seed"], how="left", indicator=True)
        old = old[old._merge == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(keys).to_csv(path, index=False)


def image_loaders(train, val, tests, kind, bg_p, textures, seed):
    if kind == "square":
        mk = lambda df, tf: SquareDataset(df, tf)
        tr = SquareDataset(train, a1_train_tf)
    else:
        mk = lambda df, tf: CropDataset(df, tf)
        tr = CropDataset(train, a1_train_tf, bg_p=bg_p, textures=textures, seed=seed)
    loaders = (DataLoader(tr, batch_size=32, shuffle=True, num_workers=0),
               DataLoader(mk(val, a1_eval_tf), batch_size=32, num_workers=0), None)
    test_loaders = {k: DataLoader(mk(v, a1_eval_tf), batch_size=32, num_workers=0) for k, v in tests.items()}
    return loaders, test_loaders


def run_image_model(name, seed, df, rec, device, textures):
    cfg = {"S1": dict(hagrid=False, bg=0.0), "S2": dict(hagrid=False, bg=0.5),
           "S3": dict(hagrid=True, bg=0.5), "S4": dict(hagrid=True, bg=0.5)}[name]
    train, val, tests = splits(df, cfg["hagrid"])
    seed_everything(seed)
    loaders, test_loaders = image_loaders(train, val, tests, "crop", cfg["bg"], textures, seed)

    def eval_all(model, model_name):
        for t, dl in test_loaders.items():
            y, p, prob = predict_proba(model, dl, device=device)
            rec.evaluate(model_name, seed, t, tests[t], y, p, prob, {"n_train": len(train)})

    with Timer() as tm:
        if name == "S4":
            model = DinoV2Classifier()
            model, hist, n_head = run_two_stage(
                model, loaders, unfreeze_dino_top, head=(6, 1e-3), full=(8, 2e-5), label=f"S4 s{seed}",
                device=device, on_head_done=lambda m: eval_all(m, "S4-probe"))
        else:
            model = resnet18_classifier()
            model, hist, n_head = run_two_stage(model, loaders, unfreeze_all, label=f"{name} s{seed}", device=device)
    print(f"  {name} s{seed}: trained on {len(train)} images in {tm.seconds:.0f}s")
    eval_all(model, name)
    rec.history(name, seed, hist, n_head)
    if not FAST:
        torch.save(model.state_dict(), MODELS_DIR / f"{name.lower()}_seed{seed}.pt")


def run_b0(seed, df, rec, device):
    _, _, tests = splits(df, False)
    model = resnet18_classifier()
    model.load_state_dict(torch.load(MODELS_DIR / f"b0_seed{seed}.pt", map_location="cpu"))
    model.to(device)
    for t, tdf in tests.items():
        dl = DataLoader(SquareDataset(tdf, a1_eval_tf), batch_size=32, num_workers=0)
        y, p, prob = predict_proba(model, dl, device=device)
        rec.evaluate("B0", seed, t, tdf, y, p, prob, {"n_train": 422})


def run_s5(seed, df, rec, device, epochs=300):
    train, val, tests = splits(df, True)
    train, val = train[train.detected].reset_index(drop=True), val[val.detected].reset_index(drop=True)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr = normalise_landmarks(landmark_array(train)), np.array([CLASS_TO_IDX[c] for c in train.label])
    xva, yva = normalise_landmarks(landmark_array(val)), np.array([CLASS_TO_IDX[c] for c in val.label])

    model = LandmarkMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    epochs = 20 if FAST else epochs
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best, best_state = -1, None
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(xtr))
        xa = augment_landmarks(xtr[perm], rng)
        tl = tc = 0
        for i in range(0, len(xa), 64):
            xb, yb = t(xa[i:i + 64]), torch.tensor(ytr[perm][i:i + 64], device=device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
        model.eval()
        with torch.no_grad():
            out = model(t(xva))
            vl = crit(out, torch.tensor(yva, device=device)).item()
            va = (out.argmax(1).cpu().numpy() == yva).mean()
        for k, v in zip(hist, (tl / len(xa), tc / len(xa), vl, va)):
            hist[k].append(v)
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  S5 s{seed}: {len(train)} train landmarks, best val acc {best:.3f}")

    model.eval()
    for name, tdf in tests.items():
        y = np.array([CLASS_TO_IDX[c] for c in tdf.label])
        p, prob = np.full(len(tdf), -1), np.zeros((len(tdf), len(CLASSES)), np.float32)
        det = tdf.detected.to_numpy(bool)
        if det.any():
            with torch.no_grad():
                pr = torch.softmax(model(t(normalise_landmarks(landmark_array(tdf[det])))), 1).cpu().numpy()
            p[det], prob[det] = pr.argmax(1), pr
        rec.evaluate("S5", seed, name, tdf, y, p, prob, {"n_train": len(train)})
    rec.history("S5", seed, hist, 0)
    if not FAST:
        torch.save(model.state_dict(), MODELS_DIR / f"s5_seed{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["B0", "S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ensure_dirs()
    device = torch.device(args.device)
    print(f"torch {torch.__version__} | device {device} | fast {FAST} | models {args.models} | seeds {args.seeds}")

    df = load_v2(fast_hands=FAST)
    train, val, tests = splits(df, True)
    print(f"train {len(train)} (own {(train.source == 'own').sum()}, hagrid {(train.source == 'hagrid').sum()})"
          f"  val {len(val)}  tests " + ", ".join(f"{k}:{len(v)}" for k, v in tests.items()))
    textures = load_textures()
    print(f"{len(textures)} background textures")

    rec = Recorder()
    for name in args.models:
        for seed in args.seeds:
            if name == "B0":
                run_b0(seed, df, rec, device)
            elif name == "S5":
                run_s5(seed, df, rec, device)
            else:
                run_image_model(name, seed, df, rec, device, textures)
            rec.save()
    print("saved", RESULTS_DIR / out_name("stage3_runs"))


if __name__ == "__main__":
    main()
