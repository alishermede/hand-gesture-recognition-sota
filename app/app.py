"""Gradio demo: hand gesture recognition (fist / peace / thumb), A1 baseline vs A2 model.

upload or webcam -> MediaPipe hand landmarks -> hand crop [-> SAM 2.1 mask] -> classifier
-> probabilities + Grad-CAM / attention rollout + per-stage latency + environment info.

Run locally:   python app/app.py            (http://127.0.0.1:7860)
HF Spaces:     this file is the Space's app.py; weights are read from WEIGHTS_DIR.
"""

import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import gradio as gr
import numpy as np
import torch

from src.config import CLASSES
from src.data import a1_eval_tf
from src.explain import attention_rollout, gradcam, overlay
from src.hands import HandDetector, centre_box, crop_square, landmarks_in_crop
from src.models import DinoV2Classifier, resnet18_classifier

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", ROOT / "results" / "a2" / "models"))
BEST = os.environ.get("BEST_MODEL", "S4")   # best on the new environments in Stage 3 (stage3_per_env.csv)
MODEL_LABELS = {"B0": "B0 — A1 baseline (ResNet-18, centre square)",
                "S3": "A2 — ResNet-18 + hand crop + bg replacement + HaGRID",
                "S4": "A2 — DINOv2 ViT-S/14 + hand crop + bg replacement + HaGRID"}
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")   # the app is measured and deployed on CPU
detector = HandDetector()
_segmenter = None
_models = {}


def segmenter():
    global _segmenter
    if _segmenter is None:
        from src.hands import HandSegmenter
        _segmenter = HandSegmenter("cpu")
    return _segmenter


def model(name):
    if name not in _models:
        if name == "S4":
            m = DinoV2Classifier(attn_implementation="eager")
        else:
            m = resnet18_classifier()
        m.load_state_dict(torch.load(WEIGHTS_DIR / f"{name.lower()}_seed42.pt", map_location="cpu"))
        _models[name] = m.eval()
    return _models[name]


def draw_hand(rgb, hand):
    out = rgb.copy()
    h, w = out.shape[:2]
    x0, y0, side = hand.box
    t = max(2, round(min(h, w) / 300))
    cv2.rectangle(out, (x0, y0), (x0 + side, y0 + side), (255, 200, 0) if hand.detected else (255, 80, 80), t)
    if hand.detected:
        pts = (hand.landmarks[:, :2] * [w, h]).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 120), t)
        for p in pts:
            cv2.circle(out, tuple(p), t + 2, (255, 255, 255), -1)
    return out


def env_info(request):
    ua = request.headers.get("user-agent", "unknown") if request is not None else "unknown"
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor() or 'unknown'})"
    return (f"**OS** {platform.system()} {platform.release()} · **Python** {platform.python_version()} · "
            f"**torch** {torch.__version__} · **inference device** CPU, {torch.get_num_threads()} threads "
            f"(available: {dev})\n\n**Browser** {ua}")


def predict(image, model_choice, use_sam, request: gr.Request):
    if image is None:
        return None, None, None, None, "Upload an image or take a webcam photo.", env_info(request)
    name = "B0" if model_choice.startswith("B0") else BEST
    rgb = np.ascontiguousarray(image[..., :3])
    times = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    hand = detector.detect(rgb)
    times["MediaPipe hand landmarks"] = time.perf_counter() - t0

    h, w = rgb.shape[:2]
    if name == "B0":   # A1 preprocessing: centre square, no hand detection used
        x0, y0, s = centre_box(w, h)
        inp = rgb[y0:y0 + s, x0:x0 + s]
    else:
        inp = crop_square(rgb, hand.box)
    vis = draw_hand(rgb, hand)

    mask_img = None
    if use_sam and hand.detected:
        t0 = time.perf_counter()
        crop = crop_square(rgb, hand.box)
        mask, _ = segmenter().segment(crop, landmarks_in_crop(hand.landmarks, w, h, hand.box))
        times["SAM 2.1 mask"] = time.perf_counter() - t0
        mask_img = crop.copy()
        mask_img[mask] = (0.5 * mask_img[mask] + 0.5 * np.array([0, 255, 120])).astype(np.uint8)

    m = model(name)
    x = a1_eval_tf(inp).unsqueeze(0)
    t0 = time.perf_counter()
    probs = torch.softmax(m(x), 1)[0].numpy()
    times["classifier"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    heat = attention_rollout(m, x) if name == "S4" else gradcam(m, x)[0]
    times["explanation (Grad-CAM / rollout)"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - t_all
    shown = cv2.resize(inp, (224, 224), interpolation=cv2.INTER_AREA)

    status = ("hand detected" if hand.detected else "**no hand detected** — centre crop used") + \
             f" · model: {MODEL_LABELS[name]}"
    lat = "| stage | ms |\n|---|---|\n" + "\n".join(f"| {k} | {1000 * v:.0f} |" for k, v in times.items())
    return (vis, {c: float(p) for c, p in zip(CLASSES, probs)}, overlay(shown, heat), mask_img,
            f"{status}\n\n{lat}", env_info(request))


ABOUT = """
### What this is
A course project (Computer Vision, Assignment 2): recognising three static hand gestures —
**fist**, **peace**, **thumb** (thumbs-up) — and comparing the Assignment 1 baseline with an
upgraded pipeline (MediaPipe hand landmarks → hand crop → classifier; SAM 2.1 masks for
background-replacement training).

### Data and licences
* Own recordings of the author's hand (and, where stated in the report, a second person's hand,
  recorded with their consent). No faces were recorded.
* **HaGRID** (Kapitanov et al.), sample `cj-mills/hagrid-sample-30k-384p`, **CC BY-SA 4.0** —
  hand crops only, split by person.
* **DTD** textures (Cimpoi et al.) for background replacement, research use.
* Models: MediaPipe Hand Landmarker (Apache-2.0), SAM 2.1 (Apache-2.0), DINOv2 (Apache-2.0),
  ResNet-18 ImageNet weights (torchvision, BSD-3).

### Privacy
Images are processed in memory and **not stored by this app**. Gradio keeps uploads in a
temporary cache, which this app deletes automatically within about 10 minutes.
Please do not upload images of other people without their consent.

### Limitations
Trained mostly on one person's right hand, indoor scenes and HaGRID crops; three gestures only.
Accuracy drops in conditions the report lists (see the per-environment table). It is not a
safety or access-control system and must not be used to identify or monitor people.

### Synthetic media
No synthetic (generated) media is used in this app.
"""


def build():
    with gr.Blocks(title="Hand gesture recognition — A1 vs A2", delete_cache=(600, 600)) as demo:
        gr.Markdown("# Hand gesture recognition: fist · peace · thumb")
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input")
                    choice = gr.Radio([MODEL_LABELS["B0"], MODEL_LABELS[BEST]], value=MODEL_LABELS[BEST],
                                      label="Model")
                    use_sam = gr.Checkbox(value=False, label="Show SAM 2.1 hand mask (slower on CPU)")
                    btn = gr.Button("Predict", variant="primary")
                with gr.Column():
                    probs = gr.Label(num_top_classes=3, label="Class probabilities")
                    info = gr.Markdown()
            with gr.Row():
                vis = gr.Image(label="Hand landmarks + crop window", type="numpy")
                cam = gr.Image(label="Grad-CAM / attention rollout on the model input", type="numpy")
                mask = gr.Image(label="SAM 2.1 mask", type="numpy")
            env = gr.Markdown()
            btn.click(predict, [img, choice, use_sam], [vis, probs, cam, mask, info, env])
        with gr.Tab("About & Ethics"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build().launch()
