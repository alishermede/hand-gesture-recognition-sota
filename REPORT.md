# Hand Gesture Recognition from a Self-Collected Dataset

**Computer Vision — Assignment 1**

Classifying three static hand gestures — **fist**, **peace** (V-sign) and **thumb**
(thumbs-up) — from images extracted from video I recorded myself.

All code and outputs are in `hand_gesture_recognition.ipynb`; every figure and number
quoted below is produced by that notebook.

---

## 1. Dataset

### 1.1 Collection

I recorded **11 videos** on a phone: 1072×1920 portrait, 60 fps, 12–26 seconds each.
Rather than shooting one long clip per gesture, I designed the recording session as a
small factorial grid over the sources of variation I could control:

| factor | levels |
|---|---|
| gesture | fist, peace, thumb |
| take (background and hand-pose set) | take 1, take 2 |
| lighting | normal, dark |

That is 3 × 2 × 2 = 12 clips. `thumb2_dark` was not recorded, so 11 exist. The two takes
use different backgrounds (a white wall and a grey upholstered headboard) and were shot
at different times, so "take" varies background, camera distance and hand orientation
together. Within each clip I continuously rotated and translated the hand, so a single
recording covers a range of viewpoints rather than one frozen pose.

### 1.2 From video to images

`build_dataset.py` converts the clips to an image dataset:

1. drop the first and last 5% of each clip, where the hand is still moving into or out
   of position;
2. sample frames at a **uniform stride** across the remainder;
3. **centre-crop to a square** and resize to **224 × 224**.

The hand is roughly centred in every clip, so a centre crop keeps it fully in frame
while avoiding the distortion a direct 1072×1920 → square resize would introduce
(verified visually across all 11 clips before committing to it).

`thumb` has one clip fewer, so it is sampled more densely — 80 frames per clip against
60 for the other classes — giving **exactly 240 images per class, 720 in total**.

| class | images | clips | dark / normal |
|---|---|---|---|
| fist | 240 | 4 | 120 / 120 |
| peace | 240 | 4 | 120 / 120 |
| thumb | 240 | 3 | 80 / 160 |

`dataset/metadata.csv` stores, for every image, its class, source clip, take, lighting
condition, frame index, position within the take, and split assignment. That bookkeeping
is what makes the evaluation protocol in §2 possible.

### 1.3 Diversity, and its limits

Figure `01_samples.png` shows two frames from every clip. The variation covered is
viewpoint and hand rotation, two backgrounds, two lighting conditions, and hand scale.
`02_brightness.png` confirms the lighting factor is real: the `_dark` clips are clearly
dimmer, **but the brightness distributions overlap across classes**, so brightness alone
carries no class information. This matters for §2.

What the dataset does *not* cover: one person, one hand, indoor scenes only, no clutter
or occlusion, no other skin tones. Those are real limits on what any result here can
claim, and I return to them in §6.

---

## 2. Splitting the data — the decision that matters most

Frames sampled from 60 fps video are **near-duplicates of their neighbours**. If they
are split randomly, almost every test image has a near-identical twin in the training
set, and the reported accuracy measures memorisation rather than recognition.

I therefore split **by clip** (a group split):

* **Test** — three entire held-out clips, `fist2_dark`, `peace2`, `thumb2` (200 images).
  Never seen in training or model selection.
* **Validation** — the last 20% (in time) of each remaining clip (98 images).
* **Train** — the first 80% of each remaining clip (422 images).

The test clips deliberately mix lighting (one dark, two normal), and the training set
retains both lighting conditions for **every** class, so neither the model nor the
evaluation can be shortcut by brightness.

### How much does this choice change the answer?

The same features and the same SVM, trained twice, differing only in the split:

| features | random frame split | held-out clip split | overstatement |
|---|---|---|---|
| raw 64×64 pixels | **0.989** | 0.295 | +0.694 |
| HOG | **0.989** | 0.245 | +0.744 |

A random frame split reports **98.9% accuracy for a model that performs below the 33.3%
chance level on a recording it has never seen**. This is the single most important
result in the project (`08_leakage.png`), and it is worth stating plainly: any gesture
recognition accuracy quoted without specifying the split protocol is unproven.

---

## 3. Preprocessing and augmentation

Two pipelines, because the classical models and the CNNs need different inputs.

**Classical models** — greyscale, resized to 64×64, then either flattened raw pixels
(4096 dims) or a **HOG descriptor** (1764 dims; 9 orientations, 8×8 cells, 2×2 blocks,
L2-Hys normalisation). HOG encodes the oriented edge structure of the fingers — the
actual signal — and its local contrast normalisation makes it far less sensitive to the
brightness shift between lighting conditions than raw pixels.

**CNNs** — 224×224 RGB (96×96 for the small CNN, to keep CPU training tractable),
normalised with ImageNet statistics so the pretrained backbone receives the input
distribution it was trained on.

**Augmentation** is applied to the training split only, and each transform is chosen to
mimic a specific way a new recording could differ from the training clips:

| transform | what it simulates |
|---|---|
| `RandomResizedCrop(scale 0.7–1.0)` | hand nearer/further, off-centre |
| `RandomRotation(±20°)` | free wrist rotation |
| `RandomHorizontalFlip` | left vs. right hand |
| `ColorJitter` (brightness, contrast, saturation, hue) | unseen lighting, white balance, skin tone |
| `RandomErasing(p=0.25)` | partial occlusion; discourages reliance on one region |

**Vertical flips are deliberately excluded**: an upside-down thumbs-up is a
thumbs-*down*, a different gesture, so that transform would corrupt the label.
`03_augmentation.png` shows the pipeline applied to a single frame.

---

## 4. Models

**Baselines**

1. Logistic Regression on raw 64×64 pixels
2. Logistic Regression on HOG
3. SVM (RBF kernel) on raw 64×64 pixels
4. SVM (RBF kernel) on HOG
5. A small CNN trained from scratch — three conv blocks with batch norm, global average
   pooling, dropout; **288,291 parameters**; 30 epochs, AdamW, cosine schedule

**Improved model**

6. **ResNet-18, ImageNet-pretrained (11.2M parameters)**, fine-tuned in two stages:
   *(i)* backbone frozen, new 3-way head trained for 6 epochs at lr 1e-3, so the randomly
   initialised head cannot destroy the pretrained features; *(ii)* everything unfrozen
   for 12 epochs at lr 5e-5. AdamW, cosine schedule, label smoothing 0.05, best-validation
   weights retained.

### Tuning the classical models honestly

Since same-clip validation cannot detect clip memorisation, I tuned the classical models
with **leave-one-clip-out cross-validation** over the 8 development clips instead:

| model | best C | LOCO-CV accuracy |
|---|---|---|
| LogReg (HOG) | 10 | 0.393 ± 0.386 |
| SVM-RBF (HOG) | 1 | 0.317 ± 0.358 |

No hyperparameter setting reaches even 0.45, and the fold-to-fold standard deviation is
enormous (~0.37) — performance depends almost entirely on *which* clip is held out.
Leave-one-clip-out CV correctly predicted the failure that the same-clip validation
split (which reported 0.86–0.93) completely hid.

---

## 5. Results

All figures are accuracy on the three **held-out clips**, 200 images never used for
training or model selection. Chance is 0.333.

| model | test accuracy | macro F1 |
|---|---|---|
| **ResNet-18 (transfer learning)** | **0.945** | **0.947** |
| Small CNN (from scratch) | 0.405 | 0.407 |
| SVM-RBF (raw pixels) | 0.295 | 0.214 |
| SVM-RBF (HOG) | 0.255 | 0.208 |
| LogReg (HOG) | 0.245 | 0.244 |
| LogReg (raw pixels) | 0.125 | 0.132 |

### Per-class performance

**ResNet-18** — macro precision 0.948, recall 0.954, F1 0.947:

| class | precision | recall | F1 |
|---|---|---|---|
| fist | 1.000 | 1.000 | 1.000 |
| peace | 0.845 | 1.000 | 0.916 |
| thumb | 1.000 | 0.863 | 0.926 |

**Small CNN** — macro F1 0.407. The aggregate hides a total collapse: fist F1 0.921,
**peace F1 0.017**, thumb F1 0.284.

**SVM-RBF (HOG)** — macro F1 0.208: fist F1 0.610, peace F1 0.015, **thumb F1 0.000**.

### Confusion structure

`06_confusion_matrices.png` shows what the baselines actually do: they assign almost
everything to one class. The SVM on HOG predicts `fist` for 83% of the true fists and
essentially never predicts `thumb` correctly. This is not random error — it is a model
that learned what the *training rooms* look like and, presented with an unfamiliar
recording, falls back on a near-constant prediction.

Per-clip accuracy makes the same point:

| model | fist2_dark | peace2 | thumb2 |
|---|---|---|---|
| ResNet-18 | 1.000 | 1.000 | 0.863 |
| Small CNN | 0.967 | 0.017 | 0.275 |
| SVM-RBF (HOG) | 0.833 | 0.017 | 0.000 |
| LogReg (HOG) | 0.667 | 0.150 | 0.000 |

Because each test clip is one gesture, per-clip accuracy is per-class recall. Every
baseline is near-perfect on exactly one clip and near-zero on the others — the signature
of a degenerate classifier, not of a model that is merely weak.

### Robustness to lighting

`fist2_dark` is the dark test clip. If a model keyed on brightness rather than shape,
accuracy there would diverge sharply from the normally-lit clips:

| model | dark clip | normal clips |
|---|---|---|
| ResNet-18 | 1.000 | 0.921 |
| Small CNN | 0.967 | 0.164 |
| SVM-RBF (HOG) | 0.833 | 0.007 |

ResNet-18 is essentially flat across lighting conditions — it is reading hand shape. The
baselines' apparent success on the dark clip is an artefact of their collapse onto the
`fist` class, which happens to be the dark clip's label.

### Training behaviour

`04_training_curves.png` shows the difference in learning dynamics. The small CNN climbs
slowly and plateaus at 0.827 validation accuracy after 30 epochs. ResNet-18 reaches 0.837
with a **frozen** backbone in 6 epochs, then jumps to 0.908 in a single epoch once
unfrozen, peaking at **0.969**. Note that validation accuracy (same-clip) overstates test
accuracy for every model — by about 2 points for ResNet-18 (0.969 vs 0.945), and by **42 points** for the small
CNN. The size of that gap is itself a measure of how much a model is memorising clips.

### Remaining errors

ResNet-18 misclassifies **11 of 200** test images, and all 11 are the same confusion:
`thumb` predicted as `peace` (`07_errors.png`). They are all very close-up frames in
which the thumb is fully extended and near-vertical while the folded fist is small,
out of focus or partly out of frame. What the model sees is an isolated upright digit
against a plain wall — geometrically much closer to the extended fingers of a peace sign
than to a canonical side-on thumbs-up, where the fist provides most of the evidence.

The confusion is strictly one-directional: no `peace` frame is ever called `thumb`. That
is consistent with a genuine loss of information at close range — the distinguishing
feature (the fist) leaves the frame — rather than with a confused decision boundary.
It also points at a concrete fix: the hand-detection-and-crop step proposed in §6 would
normalise scale and keep the whole hand in view, which is exactly what these frames lack.

---

## 6. Discussion

**Transfer learning is not a marginal improvement here — it is the difference between
working and not working.** 0.945 against 0.405 for the next-best model. With only 422
training images drawn from 8 recordings, learning good low-level filters from scratch is
the binding constraint; ImageNet's edge, texture and shape detectors are already general,
so fine-tuning only has to adapt the later blocks.

**The baselines fail for an interpretable reason.** The dataset contains only two
backgrounds, and a pixel- or HOG-space classifier has no way to separate "this is a fist"
from "this is the white-wall recording". It latches onto whichever is easier, and the
easier one is the background. HOG helps against *lighting* (it is contrast-normalised)
but not against *background*, which is why HOG and raw pixels perform similarly here.

**Validation drawn from the same recordings is actively misleading.** It reported
0.86–0.93 for models that were below chance on a new clip. Leave-one-clip-out CV caught
this; a random split did not. On grouped data, the grouping has to be respected in the
split, not just in the modelling.

**Limitations**

* One person, one hand, two backgrounds, two lighting conditions, indoor only.
* The test set is 200 images but only **three independent recordings** — the effective
  sample size for any generalisation claim is 3, not 200. Confidence in the 0.945 figure
  should be sized accordingly.
* `thumb2_dark` was never recorded, so thumb has one condition fewer and its held-out
  clip is normally lit.
* The three gestures were chosen to be visually distinct; harder pairs would be a
  stricter test.

**What I would do next**

1. **More recordings, not more frames.** Additional frames from existing clips add almost
   nothing (they are near-duplicates); additional sessions with different people, hands,
   backgrounds and distances would directly address the failure mode identified above.
2. **Detect and crop the hand first** (e.g. MediaPipe Hands), so the model sees a tight,
   consistently scaled hand region and most of the background is removed before
   classification. This is the most likely route to making the classical baselines
   viable, and would test the background-memorisation explanation directly.
3. **Add classes that differ only in finger count or orientation** (e.g. "two" vs
   "three", thumbs-up vs thumbs-down) to find where the approach breaks down.

---

## Appendix — files

| file | contents |
|---|---|
| `hand_gesture_recognition.ipynb` | all code, experiments and results, with outputs |
| `build_dataset.py` | video → image dataset |
| `dataset/`, `dataset.zip` | 720 images + `metadata.csv` |
| `figures/` | figures 01–08 referenced above |
| `results/` | metrics tables, per-class reports, trained ResNet-18 weights |
| `README.md` | how to reproduce |
