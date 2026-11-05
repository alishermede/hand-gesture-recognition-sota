"""Stage 2 helper: a small texture bank for background replacement.

Source: the DTD test split (Cimpoi et al., Describable Textures Dataset, 47 texture
classes) from the HuggingFace mirror jxie/dtd. We keep 6 random images per class
(282 textures), short side resized to 256 px. Textures are the only backgrounds used;
nothing from the A1 test clips or the new test environments is ever used as background.

Usage:  python scripts/prepare_textures.py
Writes: data_ext/textures/<class>_<i>.jpg
"""

import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pyarrow.parquet as pq

from src.config import ROOT
from src.io_utils import imwrite_unicode

URL = "https://huggingface.co/datasets/jxie/dtd/resolve/main/data/test-00000-of-00001-b49cbb6beb34aad1.parquet"
CACHE = Path.home() / "cv_a2_cache" / "dtd_test.parquet"   # outside OneDrive
OUT = ROOT / "data_ext" / "textures"
PER_CLASS = 6
SEED = 0


def main():
    if not CACHE.exists() or CACHE.stat().st_size < 170e6:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("downloading DTD test split (179 MB) ...")
        CACHE.write_bytes(urllib.request.urlopen(URL, timeout=600).read())
    table = pq.read_table(CACHE)
    df = table.to_pandas()
    img_col = next(c for c in df.columns if "image" in c.lower())
    lab_col = next(c for c in df.columns if c != img_col)
    print(f"{len(df)} DTD images, {df[lab_col].nunique()} classes, columns {list(df.columns)}")

    OUT.mkdir(parents=True, exist_ok=True)
    picks = df.groupby(lab_col, group_keys=False).sample(PER_CLASS, random_state=SEED)
    for i, r in enumerate(picks.itertuples()):
        raw = getattr(r, img_col)
        data = raw["bytes"] if isinstance(raw, dict) else raw
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        k = 256 / min(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        imwrite_unicode(OUT / f"{getattr(r, lab_col)}_{i:03d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(picks)} textures -> {OUT}")


if __name__ == "__main__":
    main()
