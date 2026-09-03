import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.evaluation.anomaly_detection import fit_gaussian, mahalanobis_scores
from src.evaluation.gradcam_eval import (
    STATS_PATH,
    load_model,
    normalize_map,
    padi_map_for_image,
)
from src.models.gradcam import GradCAMAnomaly

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "resnet18_finetuned_embeddings.pt"
THRESHOLD_PATH = PROJECT_ROOT / "results" / "anomaly_detection_comparison.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = load_model(DEVICE)
GRADCAM = GradCAMAnomaly(MODEL, str(STATS_PATH), device=DEVICE)
STATS = torch.load(STATS_PATH, map_location=DEVICE, weights_only=False)

EMBEDDINGS = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
EMBEDDING_MEAN, EMBEDDING_COV_INV = fit_gaussian(EMBEDDINGS["train_emb"])

with THRESHOLD_PATH.open() as handle:
    ANOMALY_THRESHOLD = float(json.load(handle)["finetuned"]["threshold"])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image):
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).float()


def heatmap_from_map(score_map):
    normalized = normalize_map(score_map)
    resized = cv2.resize(normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def predict(image):
    if image is None:
        return "No image provided.", None, None

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = MODEL(image_tensor).cpu()
    anomaly_score = float(mahalanobis_scores(embedding, EMBEDDING_MEAN, EMBEDDING_COV_INV)[0])
    is_anomalous = anomaly_score >= ANOMALY_THRESHOLD

    pixel_map = padi_map_for_image(MODEL, image_tensor[0], STATS)
    cam_map = GRADCAM(image_tensor)

    image_resized = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.uint8)
    gradcam_overlay = GRADCAM.overlay_on_image(image_resized, normalize_map(cam_map))
    localization_heatmap = heatmap_from_map(pixel_map)

    label = "ANOMALOUS" if is_anomalous else "OK"
    result_text = "%s  (score: %.0f, threshold: %.0f)" % (label, anomaly_score, ANOMALY_THRESHOLD)

    return result_text, localization_heatmap, gradcam_overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bottle image"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(label="PaDiM localization map"),
        gr.Image(label="Grad-CAM overlay"),
    ],
    title="Industrial Defect Inspection — Bottle Anomaly Detector",
    description="Upload a bottle image. The model flags anomalies using a fine-tuned ResNet-18 "
                "and Mahalanobis distance, with PaDiM-style localization and Grad-CAM explainability.",
    examples=[
        "tests/fixtures/sample_good.png",
        "tests/fixtures/sample_defect.png",
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )