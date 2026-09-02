import base64
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.evaluation.anomaly_detection import fit_gaussian, mahalanobis_scores
from src.evaluation.gradcam_eval import (
    MODEL_PATH,
    STATS_PATH,
    load_model,
    load_image,
    normalize_map,
    padi_map_for_image,
)
from src.models.gradcam import GradCAMAnomaly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "resnet18_finetuned_embeddings.pt"
THRESHOLD_PATH = PROJECT_ROOT / "results" / "anomaly_detection_comparison.json"

app = FastAPI(title="Anomaly Inspection API")

_model = None
_gradcam = None
_stats = None
_embedding_mean = None
_embedding_cov_inv = None
_anomaly_threshold = None
_device = None


def _load_runtime_state():
    global _model, _gradcam, _stats
    global _embedding_mean, _embedding_cov_inv, _anomaly_threshold
    global _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_model(_device)
    _gradcam = GradCAMAnomaly(_model, str(STATS_PATH), device=_device)
    _stats = torch.load(STATS_PATH, map_location=_device, weights_only=False)

    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    _embedding_mean, _embedding_cov_inv = fit_gaussian(embeddings["train_emb"])

    with THRESHOLD_PATH.open() as handle:
        _anomaly_threshold = float(json.load(handle)["finetuned"]["threshold"])


def _preprocess_upload(image):
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).float()


def _png_base64(array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError("could not encode generated image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _localization_image(score_map):
    normalized = normalize_map(score_map)
    resized = cv2.resize(normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


@app.on_event("startup")
async def startup():
    _load_runtime_state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")

    try:
        image_tensor = _preprocess_upload(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            embedding = _model(image_tensor).cpu()
        anomaly_score = float(mahalanobis_scores(embedding, _embedding_mean, _embedding_cov_inv)[0])

        pixel_map = padi_map_for_image(_model, image_tensor[0], _stats)
        cam_map = _gradcam(image_tensor)
        image_array = np.asarray(image.resize((224, 224)), dtype=np.uint8)
        overlay = _gradcam.overlay_on_image(image_array, normalize_map(cam_map))

        return {
            "anomaly_score": anomaly_score,
            "is_anomalous": anomaly_score >= _anomaly_threshold,
            "pixel_localization_map_base64": _png_base64(_localization_image(pixel_map)),
            "gradcam_overlay_base64": _png_base64(overlay),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="inference failed") from exc
