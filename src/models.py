"""Model builders and the A1 two-stage ResNet-18 fine-tuning recipe."""

import torch
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


# --- A2 models ---------------------------------------------------------------------

DINO_ID = "facebook/dinov2-small"   # ViT-S/14, 22M parameters, Apache-2.0


class DinoV2Classifier(nn.Module):
    """DINOv2 ViT-S/14 + a linear head on [CLS, mean patch token] (the DINOv2 linear-probe recipe)."""

    def __init__(self, n_classes=len(CLASSES), attn_implementation="sdpa"):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(DINO_ID, attn_implementation=attn_implementation)
        self.fc = nn.Linear(2 * self.backbone.config.hidden_size, n_classes)

    def features(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    def forward(self, x):
        return self.fc(self.features(x))


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_dino_top(model, n_blocks=4):
    """Light fine-tune: only the last n transformer blocks, the final norm and the head."""
    for blk in model.backbone.encoder.layer[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.backbone.layernorm.parameters():
        p.requires_grad = True


def run_two_stage(model, loaders, unfreeze, head=(6, 1e-3), full=(12, 5e-5), label="model",
                  fast=FAST, device=None, log=print, on_head_done=None):
    """Head-only training, then a fine-tune of whatever `unfreeze` releases (A1 recipe, generalised).
    `on_head_done(model)` is called between the stages (used to evaluate the DINOv2 linear probe)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    model, h1 = train_model(model, loaders, epochs=2 if fast else head[0], lr=head[1],
                            label=f"{label}/head", device=device, log=log)
    if on_head_done is not None:
        on_head_done(model)
    unfreeze(model)
    model, h2 = train_model(model, loaders, epochs=2 if fast else full[0], lr=full[1],
                            label=f"{label}/full", device=device, log=log)
    return model, {k: h1[k] + h2[k] for k in h1}, len(h1["train_loss"])
