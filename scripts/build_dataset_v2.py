"""Stage 1: build dataset_v2 (A1 clips + new A2 clips + HaGRID) without touching dataset/.

For every own clip the sampled frames are saved twice:
  * frames/   the full portrait frame (long side 960 px): input to the hand pipeline,
              which crops the hand from the whole camera view, as the app does;
  * square224 the A1 preprocessing (centre square, 224 px): input to B0. For the 11 A1
              clips these already exist in dataset/ and are referenced, not copied.

New clips live in raw_v2/ and are named <gesture>_<environment>_<n>.MOV
(e.g. peace_kitchen_1.MOV); thumb2_dark.MOV keeps its A1 name. All of them are
test-only. HaGRID rows come from dataset_v2/hagrid/metadata_hagrid.csv.

Usage:  python scripts/build_dataset_v2.py
Writes: dataset_v2/metadata_v2.csv, dataset_v2/frames/..., dataset_v2/square224/...
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.config import CLASSES, DATA_V2_DIR, ROOT
from src.data import load_a1_meta
from src.io_utils import imwrite_unicode

RAW_V2 = ROOT / "raw_v2"
FRAME_LONG_SIDE = 960
PER_NEW_CLIP = 60
NEW_NAME = re.compile(r"^(fist|peace|thumb)_([a-z0-9]+)_(\d+)$")
A1_NAME = re.compile(r"^(fist|peace|thumb)(\d+)(_dark)?$")


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def centre_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def downscale(frame, long_side=FRAME_LONG_SIDE):
    h, w = frame.shape[:2]
    k = long_side / max(h, w)
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame


def read_frames(video_path, indices):
    cap = cv2.VideoCapture(str(video_path))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        yield idx, (frame if ok else None)
    cap.release()


def a1_rows():
    """The 720 A1 images, plus their full frames re-read from the A1 .MOV files."""
    meta = load_a1_meta()
    rows = []
    for video, g in meta.groupby("video", sort=False):
        out_dir = DATA_V2_DIR / "frames" / video
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = dict(read_frames(ROOT / f"{video}.MOV", g.frame_index.tolist()))
        for r in g.itertuples():
            fp = out_dir / f"{video}_f{r.frame_index:05d}.jpg"
            if not fp.exists():
                imwrite_unicode(fp, downscale(frames[r.frame_index]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({
                "filepath": f"dataset/{r.filepath}", "frame_path": rel(fp), "label": r.label,
                "source": "own", "environment": "a1_room", "lighting": r.lighting, "person": "p1",
                "video": video, "frame_index": r.frame_index, "position_in_take": r.position_in_take,
                "split": r.split, "test_set": "a1_test" if r.split == "test" else "",
            })
        print(f"{video:16s} {g.label.iloc[0]:6s} {g.split.iloc[0]:6s} {len(g):3d} frames")
    return rows


def parse_new(stem):
    m = NEW_NAME.match(stem)
    if m:
        env = m.group(2)
        return m.group(1), env, "p2" if env.startswith("person") else "p1", "normal"
    m = A1_NAME.match(stem)
    if m and stem == "thumb2_dark":
        return "thumb", "a1_room_dark", "p1", "dark"
    raise ValueError(f"unexpected clip name {stem!r}: use <gesture>_<environment>_<n>.MOV")


def new_rows():
    rows = []
    videos = sorted(p for p in RAW_V2.glob("*") if p.suffix.lower() in (".mov", ".mp4")) if RAW_V2.exists() else []
    for vp in videos:
        label, env, person, lighting = parse_new(vp.stem)
        cap = cv2.VideoCapture(str(vp))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        lo, hi = int(n_frames * 0.05), int(n_frames * 0.95)  # same sampling rule as A1
        stride = (hi - lo) / PER_NEW_CLIP
        indices = [int(lo + i * stride) for i in range(PER_NEW_CLIP)]

        fdir = DATA_V2_DIR / "frames" / vp.stem
        sdir = DATA_V2_DIR / "square224" / label
        fdir.mkdir(parents=True, exist_ok=True)
        sdir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for idx, frame in read_frames(vp, indices):
            if frame is None:
                continue
            name = f"{vp.stem}_f{idx:05d}.jpg"
            imwrite_unicode(fdir / name, downscale(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
            sq = cv2.resize(centre_square(frame), (224, 224), interpolation=cv2.INTER_AREA)
            imwrite_unicode(sdir / name, sq, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({
                "filepath": rel(sdir / name), "frame_path": rel(fdir / name), "label": label,
                "source": "own", "environment": env, "lighting": lighting, "person": person,
                "video": vp.stem, "frame_index": idx,
                "position_in_take": round((idx - lo) / max(hi - lo, 1), 4),
                "split": "test", "test_set": env,
            })
            kept += 1
        print(f"{vp.stem:16s} {label:6s} {env:12s} {kept:3d} frames of {n_frames}  (test only)")
    return rows


def hagrid_rows():
    path = DATA_V2_DIR / "hagrid" / "metadata_hagrid.csv"
    if not path.exists():
        print("no HaGRID metadata: run scripts/download_hagrid.py first")
        return []
    h = pd.read_csv(path)
    return [{
        "filepath": f"dataset_v2/{r.filepath}", "frame_path": f"dataset_v2/{r.filepath}", "label": r.label,
        "source": "hagrid", "environment": "hagrid", "lighting": "", "person": r.person,
        "video": "", "frame_index": -1, "position_in_take": float("nan"),
        "split": r.split, "test_set": "hagrid_people" if r.split == "test" else "",
    } for r in h.itertuples()]


def main():
    meta = pd.DataFrame(a1_rows() + new_rows() + hagrid_rows())
    assert set(meta.label) <= set(CLASSES)
    # guards for the frozen rules: new clips and HaGRID test never in train/val
    assert (meta[meta.source == "own"].query("environment != 'a1_room'").split == "test").all()
    assert set(meta[(meta.source == "own") & (meta.split == "test") & (meta.environment == "a1_room")].video) \
        == {"fist2_dark", "peace2", "thumb2"}
    meta.to_csv(DATA_V2_DIR / "metadata_v2.csv", index=False)

    print(f"\n{len(meta)} rows -> {DATA_V2_DIR / 'metadata_v2.csv'}")
    tab = meta.pivot_table(index=["source", "split"], columns="label", values="filepath", aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(1)
    print(tab.to_string())
    tr = meta[meta.split == "train"]
    print(f"\nexternal (HaGRID) share of train: {(tr.source == 'hagrid').mean():.1%}   synthetic share: 0.0%")


if __name__ == "__main__":
    main()
