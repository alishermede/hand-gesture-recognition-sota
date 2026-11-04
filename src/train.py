"""Training loop (unchanged from A1) and prediction helpers."""

import random
import time

import numpy as np
import torch
import torch.nn as nn


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, loaders, epochs, lr, weight_decay=1e-4, label="model", device=None, log=print):
    """AdamW + cosine schedule, keeping the best-validation weights (A1 recipe)."""
    device = device or default_device()
    tr_loader, va_loader, _ = loaders
    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None

    for ep in range(epochs):
        model.train()
        tl = tn = tc = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            tn += len(yb)
        sched.step()

        model.eval()
        vl = vn = vc = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += crit(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
                vn += len(yb)

        hist["train_loss"].append(tl / tn)
        hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(vl / vn)
        hist["val_acc"].append(vc / vn)
        if vc / vn > best_acc:
            best_acc = vc / vn
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"  [{label}] epoch {ep + 1:2d}/{epochs}  "
            f"train loss {tl / tn:.3f} acc {tc / tn:.3f} | val loss {vl / vn:.3f} acc {vc / vn:.3f}"
        )

    model.load_state_dict(best_state)
    log(f"  [{label}] best val acc {best_acc:.4f}")
    return model, hist


@torch.no_grad()
def predict_proba(model, loader, device=None):
    """-> (y_true, y_pred, probabilities[n, n_classes])"""
    device = device or default_device()
    model.eval()
    probs, trues = [], []
    for xb, yb in loader:
        probs.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy())
        trues.append(yb.numpy())
    p = np.concatenate(probs)
    return np.concatenate(trues), p.argmax(1), p


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
