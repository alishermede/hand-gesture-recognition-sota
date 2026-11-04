"""Stage 1 privacy step: blur faces in the HaGRID crops (in place, before any model sees them).

HaGRID gestures are usually shown next to the face, so even the 2x hand crops often
contain a whole face. BlazeFace (short range) finds them; any detection with score >= 0.5
that is not mostly (> 60% of its area) inside the annotated hand box is blurred (the
detector also fires on knuckles, so detections on the hand itself are left alone). Faces
are blurred with a strong Gaussian over the detection box enlarged by 60%.

Run after scripts/download_hagrid.py; re-running is harmless (already-blurred faces are
no longer detected). Records the per-image count in metadata_hagrid.csv (faces_blurred).

Usage:  python scripts/blur_hagrid_faces.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.config import DATA_V2_DIR
from src.io_utils import imread_unicode, imwrite_unicode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_report import FACE_MODEL, FACE_URL  # noqa: E402

SCORE = 0.5
GROW = 0.6


def detector():
    import urllib.request

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not FACE_MODEL.exists():
        FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        FACE_MODEL.write_bytes(urllib.request.urlopen(FACE_URL, timeout=120).read())
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=FACE_MODEL.read_bytes()),
        min_detection_confidence=SCORE))
    return lambda rgb: det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).detections


def main():
    path = DATA_V2_DIR / "hagrid" / "metadata_hagrid.csv"
    meta = pd.read_csv(path)
    detect = detector()
    counts = []
    for r in meta.itertuples():
        fp = DATA_V2_DIR / r.filepath
        bgr = imread_unicode(fp)
        h, w = bgr.shape[:2]
        hx0, hy0, hx1, hy1 = r.box_x0 * w, r.box_y0 * h, r.box_x1 * w, r.box_y1 * h
        n = 0
        for d in detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
            b = d.bounding_box
            ix = max(0, min(b.origin_x + b.width, hx1) - max(b.origin_x, hx0))
            iy = max(0, min(b.origin_y + b.height, hy1) - max(b.origin_y, hy0))
            if ix * iy > 0.6 * b.width * b.height:
                continue  # a "face" on the gesture hand: detector false positive
            gx, gy = b.width * GROW / 2, b.height * GROW / 2
            x0, y0 = int(max(0, b.origin_x - gx)), int(max(0, b.origin_y - gy))
            x1, y1 = int(min(w, b.origin_x + b.width + gx)), int(min(h, b.origin_y + b.height + gy))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            k = max(15, (max(x1 - x0, y1 - y0) // 2) | 1)
            bgr[y0:y1, x0:x1] = cv2.GaussianBlur(bgr[y0:y1, x0:x1], (k, k), 0)
            n += 1
        if n:
            imwrite_unicode(fp, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts.append(n)
    meta["faces_blurred"] = np.array(counts) + (meta["faces_blurred"].fillna(0).to_numpy()
                                                 if "faces_blurred" in meta else 0)
    meta.to_csv(path, index=False)
    print(f"blurred {sum(counts)} faces in {sum(c > 0 for c in counts)} of {len(meta)} HaGRID crops")
    print(meta.assign(any=meta.faces_blurred > 0).groupby(["label", "split"])["any"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
