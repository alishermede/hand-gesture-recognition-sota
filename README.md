# Hand Gesture Recognition — Computer Vision Assignment 1

Recognising three static hand gestures — **fist**, **peace** (V-sign) and **thumb**
(thumbs-up) — from a dataset I collected myself.

## Contents

| path | what it is |
|---|---|
| `*.MOV` | the 11 raw source recordings |
| `build_dataset.py` | samples frames from the videos into the image dataset |
| `dataset/` | 720 images, 240 per class, in class folders, plus `metadata.csv` |
| `dataset.zip` | the same dataset, zipped for submission |
| `hand_gesture_recognition.ipynb` | **the main deliverable** — all code, experiments and results, with outputs |
| `notebook_src.py` | the notebook as a plain `# %%` script (what the notebook is generated from) |
| `make_notebook.py` | converts `notebook_src.py` into the `.ipynb` |
| `REPORT.md` | the written report |
| `figures/` | all figures the notebook produces |
| `results/` | metrics tables, per-class reports, trained ResNet-18 weights |

## Reproducing

```bash
pip install -r requirements.txt
python build_dataset.py                 # .MOV -> dataset/
python make_notebook.py                 # notebook_src.py -> .ipynb
jupyter nbconvert --to notebook --execute --inplace hand_gesture_recognition.ipynb
```

Everything runs on CPU. End-to-end takes roughly 20 minutes on a laptop.
Set `FAST_MODE=1` to run the script with shortened training while debugging.

## The dataset

11 phone videos (1072×1920, 60 fps, 12–26 s), covering a 3 × 2 × 2 grid of
gesture × take × lighting. `thumb2_dark` was not recorded, so there are 11 clips
rather than 12; `thumb` is sampled more densely to keep the classes balanced.

Frames are sampled at a uniform stride, centre-cropped square and resized to 224×224.
`dataset/metadata.csv` records, for every image, which clip it came from, the
lighting condition, its position within the take, and its split.

## The one thing worth knowing

The frames come from 60 fps video, so neighbouring frames are near-duplicates.
Splitting them randomly puts copies of the test images into the training set. The
notebook measures exactly what that is worth: an SVM that scores **0.99 accuracy**
under a random frame split scores **0.25 — below chance** when evaluated on a
recording it has never seen.

Every number reported here therefore comes from a **held-out-clip split**: the clips
`fist2_dark`, `peace2` and `thumb2` are never used for training or model selection.
