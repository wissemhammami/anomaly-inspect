# anomaly-inspect

Industrial anomaly detection on the MVTec AD "bottle" category: a fine-tuned ResNet-18 backbone with Mahalanobis-distance scoring, PaDiM-style pixel localization, and Grad-CAM explainability — served via a live API and a public web demo.

**Live demo:** https://wissemhammami.github.io/anomaly-inspect/
**Live API:** https://anomaly-inspect-1.onrender.com (`/health`, `/predict`, `/docs`)
**Repo:** https://github.com/wissemhammami/anomaly-inspect

---

## What it does

Given a photo of a bottle, the service returns:
- An anomaly score and a good/defective verdict
- A pixel-level localization heatmap (PaDiM-style) showing *where* the anomaly is
- A Grad-CAM overlay showing which pixels drove the model's decision

Trained and evaluated only on the MVTec AD "bottle" category — this is a scoped, single-category detector, not a general-purpose defect classifier.

---

## Pipeline

1. **Baseline CNN** — a small from-scratch CNN trained on a self-supervised rotation-prediction pretext task (no labels needed). Deliberately weak and unpretrained; kept as a documented contrast point against the ResNet-18 pipeline rather than "fixed," to make the transfer-learning ablation meaningful.
2. **ResNet-18 embeddings** — ImageNet-pretrained backbone, `fc` replaced with `Identity()`, used both frozen and fine-tuned:
   - **Frozen**: features extracted with no fine-tuning.
   - **Fine-tuned**: `layer4` + `fc` fine-tuned on the same rotation pretext task (20 epochs, ~99–100% rotation accuracy by epoch 3), then re-extracted as embeddings.
3. **Anomaly scoring** — Mahalanobis distance between a test embedding and a Gaussian fit to "good" training embeddings.
4. **Pixel localization** — PaDiM-style: a per-position (28×28) Gaussian fit on `layer2` feature maps, producing pixel-level anomaly heatmaps.
5. **Grad-CAM** — gradient-based attention map showing which pixels drove the anomaly score, compared against the PaDiM localization map for explainability.
6. **Serving** — FastAPI (`/health`, `/predict`), containerized with Docker, deployed on Render; a static HTML/CSS/JS frontend on GitHub Pages calls the API directly via `fetch()`.

---

## Results

| Metric | Result |
|---|---|
| Classification ROC-AUC (frozen embeddings, leak-free) | 0.9968 |
| Classification ROC-AUC (fine-tuned embeddings, leak-free) | 0.9952 |
| Classification accuracy (both stages) | 95.18% |
| Pixel localization ROC-AUC (PaDiM) | 0.9815 |
| Pixel localization PR-AUC (PaDiM) | 0.6837 |
| Grad-CAM↔ground-truth IoU@0.5 | ~0.01–0.15 (defect-type dependent, up to ~0.39 on `contamination`) |

Threshold calibration uses a held-out 80/20 split of the **training "good" embeddings only** (seed 42, 167 for Gaussian fitting / 42 for threshold calibration) — the 83-image test set is never touched during calibration, only for final reporting.

Pixel ROC-AUC (0.98) looks strong in isolation but is inflated by class imbalance (most pixels are non-anomalous); PR-AUC (0.68) is the more honest number and shows real remaining headroom in localization precision.

---

## Running locally

```bash
git clone https://github.com/wissemhammami/anomaly-inspect.git
cd anomaly-inspect
python -m pip install -r requirements.txt

# set MVTEC_DATA_ROOT if your data isn't at data/bottle
export MVTEC_DATA_ROOT=/path/to/mvtec/bottle

# run the API locally
uvicorn src.api.app:app --reload
# → http://localhost:8000/docs
```

Or via Docker:

```bash
docker build -t anomaly-inspect .
docker run -p 8000:8000 anomaly-inspect
```

CI (`.github/workflows/docker_build.yml`) builds the image, health-checks it, and asserts correct `/predict` behavior on known good/defect/invalid inputs on every push.

---

## Known limitations / lessons learned

- **Threshold leakage was a real early bug.** An earlier version calibrated the anomaly threshold using test-set labels, inflating reported accuracy. Fixed by calibrating only on a held-out split of training "good" embeddings — the numbers above reflect the corrected, leak-free evaluation.
- **ROC-AUC alone hid weak localization precision.** Pixel-level PaDiM localization looked strong under ROC-AUC (0.98) but PR-AUC (0.68) — the more honest metric under class imbalance — shows real room for improvement. Both are reported rather than only the flattering one.
- **Grad-CAM localization is the weakest link in the pipeline.** Even after fixing the backprop target (switched from the single max-activation pixel to a top-10%-mean over the score map, which meaningfully improved IoU-vs-ground-truth), agreement with ground-truth defect masks is still modest (~0.01–0.4 depending on defect type). This is stated here directly rather than hidden — a legitimate area for further work (e.g. Grad-CAM++, full multi-layer PaDiM).
- **PaDiM assumes spatially aligned inputs.** MVTec bottle images are well-centered so this wasn't prioritized, but it's a real constraint for deployment on less-controlled real-world images.
- **No formal three-way train/val/test split.** The test set currently does double duty for both PaDiM/localization work and final reporting; a fully separate validation set would be a more rigorous setup if this project continues to evolve.

---

## Deployment notes

- **API**: FastAPI + Docker on Render's free tier. Getting a PyTorch-based service under the 512MB free-tier RAM ceiling took several fixes: trimming unused heavy dependencies (mlflow, gradio, matplotlib, tqdm) out of the main requirements file, lazy-loading matplotlib only where needed, storing PaDiM covariance stats as float16 on disk (cast to float32 after loading, with zero accuracy impact), disabling redundant ImageNet weight downloads at container startup, and installing CPU-only PyTorch wheels.
- **Frontend**: a self-contained static page on GitHub Pages, calling the Render API directly — no separate hosting cost.
- **Gradio was evaluated and abandoned** for the live deployment: its dependency tree is too heavy for the same free tier. `Dockerfile.gradio` is kept in the repo for local-only use.

---

## License

MVTec AD dataset is licensed CC BY-NC-SA 4.0 (non-commercial use only). Code in this repository is for portfolio/educational purposes.