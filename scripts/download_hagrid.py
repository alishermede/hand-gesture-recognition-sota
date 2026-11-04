"""Stage 1: fetch a small, person-split HaGRID subset (fist / peace / like -> thumb).

Source: cj-mills/hagrid-sample-30k-384p on HuggingFace (CC BY-SA 4.0), a 384p sample
of HaGRID (Kapitanov et al.). It is one 1 GB zip with 19 classes; only the members we
need are read, over HTTP range requests (src/remote_zip.py).

Split by person: every HaGRID annotation has a `user_id`. A user goes to the held-out
test pool when sha256(user_id) % 5 == 0 (~20% of people), otherwise to the train pool,
for all classes at once, so no person is on both sides. At most one image per user
per class is kept.

Only a square crop around the gesture hand is saved (2x the hand box, so MediaPipe and
SAM 2 still have context); the full frame, which usually shows the person's face, is
not stored.

Usage:  python scripts/download_hagrid.py [--train-per-class 400] [--test-per-class 100]
Writes: dataset_v2/hagrid/<label>/<hagrid_id>.jpg
        dataset_v2/hagrid/metadata_hagrid.csv
"""

import argparse
import hashlib
import io
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.config import DATA_V2_DIR
from src.io_utils import imwrite_unicode
from src.remote_zip import HTTPRangeFile

URL = "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"
PREFIX = "hagrid-sample-30k-384p"
CLASS_MAP = {"fist": "fist", "peace": "peace", "like": "thumb"}  # HaGRID name -> our label
CONTEXT = 2.0       # crop side = CONTEXT x the longer side of the hand box
MIN_HAND_PX = 40    # skip hands smaller than this (longer box side, in pixels)
SEED = 0
OUT_DIR = DATA_V2_DIR / "hagrid"

_local = threading.local()


def zip_handle():
    if not hasattr(_local, "zip"):
        _local.zip = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL), buffer_size=1 << 16))
    return _local.zip


def is_test_person(user_id):
    return int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 5 == 0


def square_crop(img, box_px, context):
    """Square window of side context*max(w,h) centred on the box, shifted to stay inside
    the image where possible and zero-padded only if the image is too small."""
    h, w = img.shape[:2]
    x, y, bw, bh = box_px
    side = int(round(context * max(bw, bh)))
    cx, cy = x + bw / 2, y + bh / 2
    x0 = int(round(min(max(cx - side / 2, 0), max(w - side, 0))))
    y0 = int(round(min(max(cy - side / 2, 0), max(h - side, 0))))
    crop = img[y0:y0 + side, x0:x0 + side]
    pad_b, pad_r = side - crop.shape[0], side - crop.shape[1]
    if pad_b or pad_r:
        crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
    box_in_crop = ((x - x0) / side, (y - y0) / side, (x + bw - x0) / side, (y + bh - y0) / side)
    return crop, box_in_crop


def candidates(z, names):
    """One row per usable image of the three classes, with its person and hand box."""
    rows = []
    for hg, label in CLASS_MAP.items():
        ann = json.loads(z.read(f"{PREFIX}/ann_train_val/{hg}.json"))
        in_zip = {Path(n).stem: n for n in names if f"/train_val_{hg}/" in n and n.endswith(".jpg")}
        for img_id, member in in_zip.items():
            a = ann.get(img_id)
            if a is None:
                continue
            boxes = [b for b, l in zip(a["bboxes"], a["labels"]) if l == hg]
            if len(boxes) != 1:  # exactly one hand shows the gesture
                continue
            rows.append({"hagrid_id": img_id, "member": member, "hagrid_label": hg, "label": label,
                         "user_id": a["user_id"], "leading_hand": a.get("leading_hand"), "box": boxes[0]})
    return pd.DataFrame(rows)


def fetch(row):
    img = cv2.imdecode(np.frombuffer(zip_handle().read(row["member"]), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    bx, by, bw, bh = row["box"]
    box_px = (bx * w, by * h, bw * w, bh * h)
    if max(box_px[2], box_px[3]) < MIN_HAND_PX:
        return None
    crop, (x0, y0, x1, y1) = square_crop(img, box_px, CONTEXT)
    rel = f"hagrid/{row['label']}/{row['hagrid_id']}.jpg"
    imwrite_unicode(DATA_V2_DIR / rel, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"filepath": rel, "src_w": w, "src_h": h, "crop_px": crop.shape[0],
            "hand_px": round(max(box_px[2], box_px[3])),
            "box_x0": round(x0, 4), "box_y0": round(y0, 4), "box_x1": round(x1, 4), "box_y1": round(y1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=400)
    ap.add_argument("--test-per-class", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    z = zip_handle()
    cand = candidates(z, z.namelist())
    cand["split"] = np.where(cand.user_id.map(is_test_person), "test", "train")
    print(f"usable images: {len(cand)}  people: {cand.user_id.nunique()}")
    print(cand.groupby(["label", "split"]).size().unstack())

    # one image per (user, class); oversample 30% to absorb hands rejected as too small
    picks = []
    for (label, split), g in cand.groupby(["label", "split"]):
        g = g.sample(frac=1, random_state=SEED).drop_duplicates("user_id")
        want = args.train_per_class if split == "train" else args.test_per_class
        picks.append(g.head(int(want * 1.3)).assign(want=want))
    picks = pd.concat(picks).reset_index(drop=True)

    for label in CLASS_MAP.values():
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        got = list(ex.map(fetch, picks.to_dict("records")))
    picks = pd.concat([picks, pd.DataFrame([g or {} for g in got])], axis=1)
    picks = picks[picks.filepath.notna()]
    picks = picks[picks.groupby(["label", "split"]).cumcount() < picks.want]

    # files fetched as spare but not kept are removed, so the folder matches the CSV
    keep = set(picks.filepath)
    for f in OUT_DIR.glob("*/*.jpg"):
        if f"hagrid/{f.parent.name}/{f.name}" not in keep:
            f.unlink()

    meta = picks.assign(
        source="hagrid", environment="hagrid", video="", person="hagrid_" + picks.user_id.str[:12],
    )[["filepath", "label", "hagrid_label", "split", "source", "environment", "person", "hagrid_id",
       "leading_hand", "src_w", "src_h", "crop_px", "hand_px", "box_x0", "box_y0", "box_x1", "box_y1"]]
    meta.to_csv(OUT_DIR / "metadata_hagrid.csv", index=False)

    assert not set(meta[meta.split == "train"].person) & set(meta[meta.split == "test"].person)
    print("\nkept:")
    print(meta.groupby(["label", "split"]).agg(images=("filepath", "size"), people=("person", "nunique"),
                                               median_hand_px=("hand_px", "median")).to_string())
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
