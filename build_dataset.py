"""
Turn the self-recorded gesture videos into an image classification dataset.

Each .MOV is a continuous recording of one gesture under one lighting condition.
We sample frames at a uniform stride, centre-crop to a square (the hand is
roughly centred in every clip, see report) and save 224x224 JPEGs.

Output:
    dataset/<class>/<video>_f<frame>.jpg
    dataset/metadata.csv   one row per image, with the group/split bookkeeping
"""

import csv
import os
import re

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, "dataset")
IMG_SIZE = 224

# Frames to sample per video. thumb has only 3 clips (thumb2_dark was never
# recorded), so we sample it more densely to keep the classes balanced at 240.
PER_VIDEO = {"fist": 60, "peace": 60, "thumb": 80}

# One held-out clip per class: the model never sees these recordings in training.
TEST_VIDEOS = {"fist2_dark", "peace2", "thumb2"}


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite cannot handle non-ASCII paths on Windows; encode then write bytes."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img, params or [])
    if not ok:
        raise IOError(f"failed to encode {path}")
    buf.tofile(path)


def parse_name(stem):
    """'peace1_dark' -> ('peace', 1, 'dark')"""
    m = re.match(r"([a-z]+)(\d+)(_dark)?$", stem)
    if not m:
        raise ValueError(f"unexpected video name: {stem}")
    return m.group(1), int(m.group(2)), "dark" if m.group(3) else "normal"


def centre_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0 : y0 + s, x0 : x0 + s]


def main():
    videos = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith(".mov"))
    rows = []

    for video in videos:
        stem = os.path.splitext(video)[0]
        label, take, lighting = parse_name(stem)
        n_want = PER_VIDEO[label]

        cap = cv2.VideoCapture(os.path.join(SRC_DIR, video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Skip the first/last 5%: the hand is often still moving into or out of
        # position at the very start and end of a take.
        lo, hi = int(n_frames * 0.05), int(n_frames * 0.95)
        stride = (hi - lo) / n_want
        indices = [int(lo + i * stride) for i in range(n_want)]

        cls_dir = os.path.join(OUT_DIR, label)
        os.makedirs(cls_dir, exist_ok=True)

        kept = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            img = cv2.resize(
                centre_square(frame), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            fname = f"{stem}_f{idx:05d}.jpg"
            imwrite_unicode(
                os.path.join(cls_dir, fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            rows.append(
                {
                    "filepath": f"{label}/{fname}",
                    "label": label,
                    "video": stem,
                    "take": take,
                    "lighting": lighting,
                    "frame_index": idx,
                    "position_in_take": round((idx - lo) / max(hi - lo, 1), 4),
                    "split": "test" if stem in TEST_VIDEOS else "trainval",
                }
            )
            kept += 1
        cap.release()
        print(f"{stem:16s} {label:6s} {lighting:6s} {kept:4d} frames of {n_frames}")

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\ntotal {len(rows)} images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
