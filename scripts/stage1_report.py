"""Stage 1 report: dataset table, face check and contact sheets for dataset_v2.

  * dataset table: class x source x split, with the external (HaGRID) and synthetic share
  * face check: MediaPipe Face Detector (BlazeFace short range) on every own frame and
    every HaGRID crop (after scripts/blur_hagrid_faces.py); score >= 0.8 is flagged for review
  * contact sheets: one frame per own clip, a random HaGRID sample, and every flagged image

Usage:  python scripts/stage1_report.py
Writes: results/a2/stage1_dataset_table.csv, stage1_face_flags.csv,
        figures/a2/stage1_contact_own.png, stage1_contact_hagrid.png, stage1_faces_flagged.png
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASSES, DATA_V2_DIR, FIG_DIR, RESULTS_DIR, ROOT, ensure_dirs
from src.io_utils import load_rgb

FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
FACE_MODEL = ROOT / "cache" / "blaze_face_short_range.tflite"
# Calibrated on the own frames, which contain no faces (checked by eye): BlazeFace fires on
# knuckles there with scores up to 0.79, so only detections above 0.8 are flagged.
FACE_THRESHOLD = 0.8


def face_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not FACE_MODEL.exists():
        FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        FACE_MODEL.write_bytes(urllib.request.urlopen(FACE_URL, timeout=120).read())
    opts = vision.FaceDetectorOptions(base_options=mp_python.BaseOptions(model_asset_buffer=FACE_MODEL.read_bytes()),
                                      min_detection_confidence=0.3)
    det = vision.FaceDetector.create_from_options(opts)
    return lambda rgb: [d.categories[0].score for d in
                        det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).detections]


def dataset_table(meta):
    tab = (meta.assign(n=1).pivot_table(index=["source", "split"], columns="label", values="n",
                                        aggfunc="sum", fill_value=0).reindex(columns=CLASSES, fill_value=0))
    tab["total"] = tab.sum(axis=1)
    tab = tab.reset_index()
    tr = meta[meta.split == "train"]
    share = {"external_share_of_train": round((tr.source == "hagrid").mean(), 4),
             "external_share_of_all": round((meta.source == "hagrid").mean(), 4),
             "synthetic_share": 0.0}
    return tab, share


def sheet(rows, title, path, ncol=10, flag=None):
    n = len(rows)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.6 * ncol, 1.9 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows.itertuples()):
        ax.imshow(load_rgb(ROOT / r.frame_path))
        bad = flag is not None and r.filepath in flag
        ax.set_title(f"{r.label}\n{r.tag}" + ("\nFACE?" if bad else ""), fontsize=6,
                     color="red" if bad else "black")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    ensure_dirs()
    meta = pd.read_csv(DATA_V2_DIR / "metadata_v2.csv")
    tab, share = dataset_table(meta)
    tab.to_csv(RESULTS_DIR / "stage1_dataset_table.csv", index=False)
    print(tab.to_string(index=False))
    print(share)
    pd.DataFrame([share]).to_csv(RESULTS_DIR / "stage1_shares.csv", index=False)

    detect = face_detector()
    flags = []
    for r in meta.itertuples():
        scores = detect(load_rgb(ROOT / r.frame_path))
        flags.append({"filepath": r.filepath, "source": r.source, "n_faces": len(scores),
                      "max_face_score": round(max(scores), 3) if scores else 0.0})
    flags = pd.DataFrame(flags)
    flags["flagged"] = flags.max_face_score >= FACE_THRESHOLD
    flags.to_csv(RESULTS_DIR / "stage1_face_flags.csv", index=False)
    print(flags.groupby("source").agg(images=("filepath", "size"), flagged=("flagged", "sum")).to_string())
    flagged = set(flags[flags.flagged].filepath)

    own = meta[meta.source == "own"]
    one_per_clip = own.groupby("video").nth(30).reset_index(drop=True)
    one_per_clip["tag"] = one_per_clip.video + "\n" + one_per_clip.split
    sheet(one_per_clip, "Own clips: one frame per clip (A1 + A2)", FIG_DIR / "stage1_contact_own.png", flag=flagged)

    hg = meta[meta.source == "hagrid"].groupby(["label", "split"], group_keys=False).sample(5, random_state=0)
    hg["tag"] = hg.split + " " + hg.person.str[-6:]
    sheet(hg, "HaGRID crops (random sample, person-split)", FIG_DIR / "stage1_contact_hagrid.png", flag=flagged)

    fl = meta[meta.filepath.isin(flagged)].head(60).copy()
    fl["tag"] = fl.source
    sheet(fl, f"Images flagged by the face detector (score >= {FACE_THRESHOLD}), first 60",
          FIG_DIR / "stage1_faces_flagged.png", flag=flagged)


if __name__ == "__main__":
    main()
