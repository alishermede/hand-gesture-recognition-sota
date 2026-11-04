"""Explainability: Grad-CAM (ResNet-18), attention rollout (DINOv2), and how much of a heatmap
falls on the hand.

`energy_on_mask` turns "is the model looking at the hand or the background?" into a
number: the share of the heatmap's mass inside the SAM 2.1 hand mask. The mask's own
area is the chance level (a heatmap spread uniformly over the image).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x, target=None, layer=None):
    """Grad-CAM on `layer` (default: ResNet layer4). x: (1, 3, H, W). -> (cam HxW in [0,1], class)"""
    layer = layer or model.layer4
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    model.eval()
    try:
        with torch.enable_grad():
            out = model(x)
            cls = int(out.argmax(1)) if target is None else int(target)
            model.zero_grad()
            out[0, cls].backward()
    finally:
        h1.remove()
        h2.remove()
    w = store["grad"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam, cls


@torch.no_grad()
def attention_rollout(dino_model, x, discard_ratio=0.0):
    """Attention rollout (Abnar & Zuidema, 2020) from the CLS token to the patches.
    `dino_model` must be a DinoV2Classifier built with attn_implementation="eager"."""
    out = dino_model.backbone(pixel_values=x, output_attentions=True)
    n_tok = out.attentions[0].shape[-1]
    roll = torch.eye(n_tok, device=x.device)
    for att in out.attentions:
        a = att[0].mean(0)                      # average heads
        if discard_ratio:
            flat = a.flatten()
            k = int(flat.numel() * discard_ratio)
            flat[flat.topk(k, largest=False).indices] = 0
        a = a + torch.eye(n_tok, device=x.device)
        a = a / a.sum(-1, keepdim=True)
        roll = a @ roll
    cls_to_patch = roll[0, 1:]
    side = int(cls_to_patch.numel() ** 0.5)
    m = cls_to_patch.reshape(1, 1, side, side)
    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
    return m


def energy_on_mask(heat, mask):
    """Share of the heatmap mass inside the mask; the mask area fraction is the chance level."""
    mask = cv2.resize(mask.astype(np.uint8), heat.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    total = heat.sum()
    return float(heat[mask].sum() / total) if total > 0 else float("nan"), float(mask.mean())


def overlay(rgb, heat, alpha=0.45):
    """Jet-coloured heatmap over an RGB image (both HxW)."""
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    col = cv2.applyColorMap((255 * heat).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (alpha * col + (1 - alpha) * rgb).astype(np.uint8)
