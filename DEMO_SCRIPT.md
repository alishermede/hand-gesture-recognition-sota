# Demo video — shot list (3–5 min)

Record the screen (OBS / Xbox Game Bar `Win+Alt+R`) at 1080p with the microphone on.
Before recording: `app/app.py` running locally, the browser at `http://127.0.0.1:7860`,
`REPORT_2.md` (or the PDF) and `a2_experiments.ipynb` open in other tabs. Good light, plain
background for the first live shots, then a cluttered one. **No faces in the webcam frame.**
Every number you say must be read off the report tables, not from memory.

| # | time | on screen | say (key points) |
|---|---|---|---|
| 1 | 0:00–0:20 | Title slide or the report's first page | Topic from A1: 3 static gestures (fist, peace, thumb). A2 goal: make it work outside my room, show why, and ship a demo. |
| 2 | 0:20–0:50 | A1 recap: `figures/a2/stage0_errors_fullframe.png` | A1 baseline reproduced exactly (same seed, same numbers); over 3 seeds it is lower than the single A1 run. All errors are close-up thumbs on one clip; the model only saw the centre square. |
| 3 | 0:50–1:30 | `figures/a2/pipeline_diagram.png` | The SOTA components: MediaPipe Hand Landmarker finds 21 landmarks → square hand crop; SAM 2.1 turns the landmarks into a hand mask → background replacement during training; DINOv2 as a modern backbone; HaGRID (other people, split by person). Why not Step-Video-T2V / ArtAug (30B / needs ~77 GB VRAM; ArtAug is an aesthetic T2I LoRA). |
| 4 | 1:30–2:30 | App, **Predict** tab, webcam | Show fist → peace → thumb with the A2 model: landmarks + crop window, probabilities, Grad-CAM on the hand, latency breakdown and environment line. Then a **close-up thumb** (the A1 failure case). |
| 5 | 2:30–3:00 | Same app, switch the selector to **B0** | Same gestures on a cluttered background: B0 vs A2 side by side; point at the Grad-CAM (hand vs background). Tick "Show SAM 2.1 mask" once and read the extra latency. |
| 6 | 3:00–3:50 | Report: per-environment table, ablation table, latency table | Accuracy per environment for B0 vs best model; what each step added (crop / background / HaGRID); CPU latency per component and end to end. Mention mean ± std over 3 seeds. |
| 7 | 3:50–4:20 | Report: explainability figure (B0 vs S2 heatmaps) | Share of the heatmap on the hand vs on the background, B0 vs A2. |
| 8 | 4:20–4:50 | App, **About & Ethics** tab | Consent (own hand; second person with consent), HaGRID licence and face blurring, "images not stored", limitations, no synthetic media used (or: synthetic stress test labelled SYNTHETIC). |
| 9 | 4:50–5:00 | Repo / README | How to reproduce: one venv, `scripts/` stages, notebook runs top to bottom. |

Backup if the webcam misbehaves during the defense: upload frames from `dataset_v2/frames/`
of the new-environment clips (test-only data, never trained on).
