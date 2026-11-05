"""Stage 3: CPU latency at batch size 1, per component and end to end, plus model size.

Components: MediaPipe Hand Landmarker, SAM 2.1 (hiera-tiny, on the 256 px crop),
ResNet-18 / DINOv2 ViT-S/14 / landmark MLP classifiers. End-to-end pipelines start
from a decoded RGB frame (540x960) and end with class probabilities.

Usage:  python scripts/stage3_latency.py [--runs 50] [--frames 20]
Writes: results/a2/stage3_latency.csv, results/a2/stage3_model_size.csv
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.config import DATA_V2_DIR, MODELS_DIR, RESULTS_DIR, ROOT, ensure_dirs, out_name
from src.crops import LandmarkMLP, normalise_landmarks
from src.data import a1_eval_tf
from src.hands import HandDetector, HandSegmenter, centre_box, crop_square, landmarks_in_crop
from src.io_utils import load_rgb
from src.models import DinoV2Classifier, resnet18_classifier


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def timeit(fn, inputs, runs, warmup=5):
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    ts = []
    for i in range(runs):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        fn(x)
        ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    ensure_dirs()
    torch.set_grad_enabled(False)
    dev = torch.device("cpu")

    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")
    own = meta[(meta.source == "own") & (meta.split == "test")].sample(args.frames, random_state=0)
    frames = [load_rgb(ROOT / p) for p in own.frame_path]

    det = HandDetector()
    seg = HandSegmenter("cpu")
    resnet = resnet18_classifier().eval()
    dino = DinoV2Classifier().eval()
    mlp = LandmarkMLP().eval()
    for m, f in [(resnet, "s3_seed42.pt"), (dino, "s4_seed42.pt"), (mlp, "s5_seed42.pt")]:
        if (MODELS_DIR / f).exists():
            m.load_state_dict(torch.load(MODELS_DIR / f, map_location="cpu"))

    hands = [det.detect(f) for f in frames]
    crops = [crop_square(f, h.box) for f, h in zip(frames, hands)]
    found = [(c, landmarks_in_crop(h.landmarks, f.shape[1], f.shape[0], h.box))
             for f, h, c in zip(frames, hands, crops) if h.detected]
    tensors = [a1_eval_tf(c).unsqueeze(0) for c in crops]
    lms = [torch.tensor(normalise_landmarks(h.landmarks[None] * [f.shape[1], f.shape[0], f.shape[1]]),
                        dtype=torch.float32) for f, h in zip(frames, hands) if h.detected]

    def e2e_b0(f):
        x0, y0, s = centre_box(f.shape[1], f.shape[0])
        return resnet(a1_eval_tf(f[y0:y0 + s, x0:x0 + s]).unsqueeze(0))

    def e2e_crop(model):
        def run(f):
            h = det.detect(f)
            return model(a1_eval_tf(crop_square(f, h.box)).unsqueeze(0))
        return run

    def e2e_s5(f):
        h = det.detect(f)
        if h.detected:
            return mlp(torch.tensor(normalise_landmarks(h.landmarks[None] * [f.shape[1], f.shape[0], f.shape[1]]),
                                    dtype=torch.float32))

    def e2e_sam(f):
        h = det.detect(f)
        c = crop_square(f, h.box)
        if h.detected:
            seg.segment(c, landmarks_in_crop(h.landmarks, f.shape[1], f.shape[0], h.box))
        return resnet(a1_eval_tf(c).unsqueeze(0))

    tests = [
        ("component", "MediaPipe Hand Landmarker", lambda f: det.detect(f), frames),
        ("component", "SAM 2.1 hiera-tiny (256 px crop)", lambda cp: seg.segment(*cp), found),
        ("component", "ResNet-18 classifier (224 px)", lambda x: resnet(x), tensors),
        ("component", "DINOv2 ViT-S/14 classifier (224 px)", lambda x: dino(x), tensors),
        ("component", "Landmark MLP", lambda x: mlp(x), lms),
        ("end-to-end", "B0: centre square + ResNet-18", e2e_b0, frames),
        ("end-to-end", "S3: MediaPipe + crop + ResNet-18", e2e_crop(resnet), frames),
        ("end-to-end", "S4: MediaPipe + crop + DINOv2", e2e_crop(dino), frames),
        ("end-to-end", "S5: MediaPipe + landmark MLP", e2e_s5, frames),
        ("end-to-end", "S3 + SAM 2.1 mask overlay (app option)", e2e_sam, frames),
    ]
    env = {"cpu": cpu_name(), "threads": torch.get_num_threads(), "torch": torch.__version__,
           "python": platform.python_version(), "os": f"{platform.system()} {platform.release()}"}
    rows = []
    for kind, name, fn, inputs in tests:
        med, p95 = timeit(fn, inputs, args.runs)
        rows.append({"kind": kind, "name": name, "median_ms": round(med, 1), "p95_ms": round(p95, 1),
                     "fps_median": round(1000 / med, 1), "runs": args.runs, **env})
        print(f"{kind:10s} {name:42s} median {med:7.1f} ms   p95 {p95:7.1f} ms")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / out_name("stage3_latency"), index=False)

    sizes = []
    for name, m, f in [("ResNet-18 (B0-S3)", resnet, "s3_seed42.pt"), ("DINOv2 ViT-S/14 + head (S4)", dino, "s4_seed42.pt"),
                       ("Landmark MLP (S5)", mlp, "s5_seed42.pt"), ("SAM 2.1 hiera-tiny", seg.model, None)]:
        n = sum(p.numel() for p in m.parameters())
        mb = (MODELS_DIR / f).stat().st_size / 1e6 if f and (MODELS_DIR / f).exists() else 4 * n / 1e6
        sizes.append({"model": name, "parameters": n, "size_mb": round(mb, 1)})
    mp_size = (ROOT / "cache" / "hand_landmarker.task").stat().st_size / 1e6
    sizes.append({"model": "MediaPipe Hand Landmarker (.task)", "parameters": np.nan, "size_mb": round(mp_size, 1)})
    pd.DataFrame(sizes).to_csv(RESULTS_DIR / out_name("stage3_model_size"), index=False)
    print(pd.DataFrame(sizes).to_string(index=False))
    print(env)


if __name__ == "__main__":
    main()
