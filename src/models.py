"""Model builders and the A1 two-stage ResNet-18 fine-tuning recipe."""

import torch.nn as nn
import torchvision

from .config import CLASSES, FAST
from .data import make_loaders
from .train import seed_everything, train_model


def resnet18_classifier(n_classes=len(CLASSES)):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def run_resnet18(train_df, val_df, test_df, train_tf, eval_tf, seed, fast=FAST, device=None, log=print):
    """The A1 recipe, end to end: head only (6 ep, lr 1e-3), then full fine-tune (12 ep, lr 5e-5).

    Seeding, loader construction and model construction happen here, in the same order
    as in A1, so seed 42 consumes the RNG exactly as the A1 notebook did.
    Returns (model, history, loaders, n_head_epochs).
    """
    seed_everything(seed)
    loaders = make_loaders(train_df, val_df, test_df, train_tf, eval_tf, bs=32)
    model = resnet18_classifier()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    model, h1 = train_model(model, loaders, epochs=2 if fast else 6, lr=1e-3,
                            label=f"ResNet18/head s{seed}", device=device, log=log)

    for p in model.parameters():
        p.requires_grad = True
    model, h2 = train_model(model, loaders, epochs=2 if fast else 12, lr=5e-5, weight_decay=1e-4,
                            label=f"ResNet18/full s{seed}", device=device, log=log)

    hist = {k: h1[k] + h2[k] for k in h1}
    return model, hist, loaders, len(h1["train_loss"])
