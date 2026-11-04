"""Stage 2: cache MediaPipe landmarks, hand crops and SAM 2.1 masks for dataset_v2.

For every row of dataset_v2/metadata_v2.csv:
  * MediaPipe Hand Landmarker on the full frame (largest hand kept);
  * square crop with 25% margin around the landmarks, or the A1 centre square when no
    hand is found (logged: `detected` = False);
  * SAM 2.1 mask of the crop, prompted with the landmark bounding box — only for
    training rows (background replacement is train-only) and rows with a hand.

Usage:  python scripts/precompute_hands.py [--device cuda] [--no-sam] [--limit N]
        FAST_MODE=1 ...   processes 40 rows only
Writes: dataset_v2/hands.csv         one row per image (landmarks, box, scores, paths)
        dataset_v2/crops/<...>.jpg   256x256 hand crops
        dataset_v2/masks/<...>.png   256x256 SAM masks (train rows)
        results/a2/stage2_detection.csv   detection rate per source / environment / split
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.config import DATA_V2_DIR, FAST, RESULTS_DIR, ROOT, ensure_dirs, out_name
from src.hands import (HandDetector, HandSegmenter, crop_square, frac_outside_centre_square,
                       landmarks_in_crop)
from src.io_utils import imwrite_unicode, load_rgb


def cache_name(filepath, ext):
    return Path(filepath).with_suffix(ext).as_posix().replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()

    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")
    if FAST or args.limit:
        meta = meta.groupby(["source", "split"], group_keys=False).head(args.limit or 8)
    crop_dir, mask_dir = DATA_V2_DIR / "crops", DATA_V2_DIR / "masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    det = HandDetector()
    seg = None if args.no_sam else HandSegmenter(args.device)
    rows, t_mp, t_sam = [], [], []
    for i, r in enumerate(meta.itertuples()):
        rgb = load_rgb(ROOT / r.frame_path)
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        hand = det.detect(rgb)
        t_mp.append(time.perf_counter() - t0)

        crop = crop_square(rgb, hand.box)
        crop_rel = f"dataset_v2/crops/{cache_name(r.filepath, '.jpg')}"
        imwrite_unicode(ROOT / crop_rel, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        row = {"filepath": r.filepath, "detected": hand.detected, "n_hands": hand.n_hands,
               "score": round(hand.score, 4), "handedness": hand.handedness,
               "frame_w": w, "frame_h": h, "box_x0": hand.box[0], "box_y0": hand.box[1], "box_side": hand.box[2],
               "crop_path": crop_rel, "mask_path": "", "mask_area": np.nan, "sam_iou": np.nan,
               "frac_outside_centre": np.nan}
        if hand.detected:
            row["frac_outside_centre"] = round(frac_outside_centre_square(hand.landmarks, w, h), 4)
            pts = landmarks_in_crop(hand.landmarks, w, h, hand.box)
            for j in range(21):
                row[f"x{j}"], row[f"y{j}"], row[f"z{j}"] = (round(float(v), 5) for v in hand.landmarks[j])
            if seg is not None and r.split == "train":
                t0 = time.perf_counter()
                mask, iou = seg.segment(crop, pts)
                t_sam.append(time.perf_counter() - t0)
                mask_rel = f"dataset_v2/masks/{cache_name(r.filepath, '.png')}"
                imwrite_unicode(ROOT / mask_rel, mask.astype(np.uint8) * 255)
                row.update(mask_path=mask_rel, mask_area=round(float(mask.mean()), 4), sam_iou=round(iou, 4))
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(meta)}  detected so far {np.mean([x['detected'] for x in rows]):.1%}")

    hands = pd.DataFrame(rows)
    hands.to_csv(DATA_V2_DIR / out_name("hands"), index=False)

    m = meta.merge(hands[["filepath", "detected", "mask_path"]], on="filepath")
    m["has_mask"] = m.mask_path.fillna("").str.len() > 0
    rate = (m.groupby(["source", "environment", "split"])
             .agg(images=("filepath", "size"), detected=("detected", "mean"), masks=("has_mask", "sum"))
             .reset_index())
    rate.to_csv(RESULTS_DIR / out_name("stage2_detection"), index=False)
    print(rate.round(3).to_string(index=False))
    print(f"\noverall detection rate {m.detected.mean():.1%} over {len(m)} images")
    print(f"MediaPipe {1000 * np.median(t_mp):.1f} ms/img (median)"
          + (f" | SAM 2.1 {1000 * np.median(t_sam):.1f} ms/img on {args.device}" if t_sam else ""))


if __name__ == "__main__":
    main()
