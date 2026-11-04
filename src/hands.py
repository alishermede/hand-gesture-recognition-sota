"""Hand pipeline: MediaPipe Hand Landmarker -> square hand crop -> SAM 2.1 mask -> background replacement.

Used identically by scripts/precompute_hands.py (cached for training), the evaluation
code and the Gradio app, so train and test see the same crops.
"""

import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import ROOT

MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task")
MP_MODEL_PATH = ROOT / "cache" / "hand_landmarker.task"
SAM_ID = "facebook/sam2.1-hiera-tiny"

CROP_MARGIN = 0.25    # margin on each side of the landmark box -> crop side = 1.5 x box
CROP_SIZE = 256       # cached crops / masks are stored at this resolution
MIN_DETECT_SIDE = 256 # small images (HaGRID crops) are upscaled to this before detection

# landmarks used as positive point prompts for SAM 2: wrist, the four MCP knuckles,
# and the thumb / index / middle tips (they lie on the hand for all three gestures)
SAM_POINT_IDS = (0, 5, 9, 13, 17, 4, 8, 12)


@dataclass
class HandResult:
    detected: bool
    landmarks: np.ndarray | None = None   # (21, 3), x/y normalised to the image, z relative
    score: float = 0.0
    handedness: str = ""
    n_hands: int = 0
    box: tuple = field(default_factory=tuple)  # crop window (x0, y0, side) in image pixels


# --- MediaPipe ---------------------------------------------------------------------

class HandDetector:
    def __init__(self, min_conf=0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not MP_MODEL_PATH.exists():
            MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            MP_MODEL_PATH.write_bytes(urllib.request.urlopen(MP_MODEL_URL, timeout=120).read())
        # MediaPipe cannot open non-ASCII paths on Windows: hand it the model bytes instead.
        base = mp_python.BaseOptions(model_asset_buffer=MP_MODEL_PATH.read_bytes())
        opts = vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
        )
        self._mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb):
        """Largest detected hand in an RGB uint8 image, plus its square crop window."""
        h, w = rgb.shape[:2]
        k = max(1.0, MIN_DETECT_SIDE / min(h, w))
        img = cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_CUBIC) if k > 1 else rgb
        res = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                    data=np.ascontiguousarray(img)))
        if not res.hand_landmarks:
            return HandResult(False, box=centre_box(w, h))

        hands = [np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32) for lm in res.hand_landmarks]
        areas = [np.ptp(l[:, 0]) * w * np.ptp(l[:, 1]) * h for l in hands]
        i = int(np.argmax(areas))
        cat = res.handedness[i][0]
        lm = hands[i]
        return HandResult(True, lm, float(cat.score), cat.category_name, len(hands), crop_box(lm, w, h))


def crop_box(lm, w, h, margin=CROP_MARGIN):
    """Square window around the landmarks, 'margin' of the box size added on each side."""
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    side = max(np.ptp(xs), np.ptp(ys)) * (1 + 2 * margin)
    side = max(side, 32.0)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return (int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side)))


def centre_box(w, h):
    """Fallback when no hand is found: the A1 centre square."""
    s = min(w, h)
    return ((w - s) // 2, (h - s) // 2, s)


def crop_square(img, box, size=CROP_SIZE):
    """Cut the (x0, y0, side) window, zero-padding where it leaves the image, and resize."""
    x0, y0, side = box
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    crop = img[y0:y0 + side, x0:x0 + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interp)


def landmarks_in_crop(lm, w, h, box, size=CROP_SIZE):
    """Landmark pixel coordinates inside the resized crop."""
    x0, y0, side = box
    return np.stack([(lm[:, 0] * w - x0) / side * size, (lm[:, 1] * h - y0) / side * size], 1)


def frac_outside_centre_square(lm, w, h):
    """Share of the 21 landmarks that fall outside the A1 centre square (close-up analysis)."""
    x0, y0, s = centre_box(w, h)
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    return float(((xs < x0) | (xs >= x0 + s) | (ys < y0) | (ys >= y0 + s)).mean())


# --- SAM 2.1 -----------------------------------------------------------------------

class HandSegmenter:
    """SAM 2.1 (hiera-tiny) through transformers, prompted with landmark points + their box."""

    def __init__(self, device="cpu"):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch, self.device = torch, device
        self.processor = Sam2Processor.from_pretrained(SAM_ID)
        self.model = Sam2Model.from_pretrained(SAM_ID).to(device).eval()

    def segment(self, crop_rgb, pts):
        """crop_rgb: (S, S, 3) uint8; pts: (21, 2) landmark pixels in the crop. -> (mask bool, iou score)"""
        s = crop_rgb.shape[0]
        sel = pts[list(SAM_POINT_IDS)]
        sel = sel[(sel[:, 0] >= 0) & (sel[:, 0] < s) & (sel[:, 1] >= 0) & (sel[:, 1] < s)]
        if len(sel) == 0:
            return np.zeros((s, s), bool), 0.0
        x0, y0 = np.clip(pts.min(0) - 0.08 * s, 0, s - 1)
        x1, y1 = np.clip(pts.max(0) + 0.08 * s, 0, s - 1)
        inputs = self.processor(
            images=crop_rgb,
            input_points=[[sel.round().astype(int).tolist()]],
            input_labels=[[[1] * len(sel)]],
            input_boxes=[[[float(x0), float(y0), float(x1), float(y1)]]],
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        mask = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        return mask[0, 0].numpy().astype(bool), float(out.iou_scores.flatten()[0])


# --- background replacement (train only) -------------------------------------------

def random_gradient(size, rng):
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    t = (np.cos(ang) * xx + np.sin(ang) * yy)
    t = (t - t.min()) / max(np.ptp(t), 1e-6)
    return (c1 * (1 - t[..., None]) + c2 * t[..., None]).astype(np.uint8)


def replace_background(crop_rgb, mask, textures, rng, feather=2.0):
    """Composite the masked hand onto a DTD texture (or a random gradient, 50/50)."""
    s = crop_rgb.shape[0]
    if textures and rng.random() < 0.5:
        tex = textures[rng.integers(len(textures))]
        th, tw = tex.shape[:2]
        side = int(rng.integers(min(th, tw) // 2, min(th, tw) + 1))
        ty, tx = rng.integers(0, th - side + 1), rng.integers(0, tw - side + 1)
        bg = cv2.resize(tex[ty:ty + side, tx:tx + side], (s, s), interpolation=cv2.INTER_AREA)
    else:
        bg = random_gradient(s, rng)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None]
    return (alpha * crop_rgb + (1 - alpha) * bg).astype(np.uint8)
