import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from src.models.gradcam import GradCAMAnomaly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "bottle"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_finetuned.pt"
STATS_PATH = PROJECT_ROOT / "models" / "localization_stats.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "gradcam_examples"


def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def load_mask(path):
    if path is None or not path.exists():
        return np.zeros((224, 224), dtype=np.uint8)
    mask = Image.open(path).convert("L").resize((224, 224))
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.uint8)


def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    denom = arr.max() - arr.min()
    if denom < 1e-8:
        return np.zeros_like(arr)
    return arr / denom


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def iou_at_threshold(a, b, threshold=0.5):
    a = np.asarray(a, dtype=np.float32) >= threshold
    b = np.asarray(b, dtype=np.float32) >= threshold
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def collect_sample_paths():
    test_dir = DATA_ROOT / "test"
    folders = sorted(os.listdir(test_dir))
    print("sorted test folders:", folders)

    good_paths = sorted((test_dir / "good").glob("*.png"))
    defect_paths = []
    for folder_name in folders:
        folder = test_dir / folder_name
        if not folder.is_dir() or folder_name == "good":
            continue
        defect_paths.extend(sorted(folder.glob("*.png")))

    sample_paths = good_paths + defect_paths
    print("selected sample paths:")
    for p in sample_paths:
        print("  ", p.relative_to(PROJECT_ROOT))
    return sample_paths


def load_model(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    filtered = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.load_state_dict(filtered, strict=False)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    return model


def padi_map_for_image(model, image_tensor, stats):
    x = image_tensor.unsqueeze(0).to(image_tensor.device)
    with torch.no_grad():
        out = model.conv1(x)
        out = model.bn1(out)
        out = model.relu(out)
        out = model.maxpool(out)
        out = model.layer1(out)
        out = model.layer2(out)

    feat = out[0].permute(1, 2, 0).reshape(-1, out.shape[1]).cpu().numpy()
    mean = stats["mean"]
    cov_inv = stats["cov_inv"]
    scores = np.zeros(feat.shape[0], dtype=np.float32)
    for pos in range(feat.shape[0]):
        diff = feat[pos] - mean[pos]
        scores[pos] = diff @ cov_inv[pos] @ diff
    return scores.reshape(28, 28)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    gradcam = GradCAMAnomaly(model, str(STATS_PATH), device=device)
    stats = torch.load(STATS_PATH, map_location=device, weights_only=False)
    sample_paths = collect_sample_paths()

    agreement = []

    for path in sample_paths:
        img = Image.open(path).convert("RGB")
        img_arr = np.asarray(img)
        x = load_image(path).unsqueeze(0).to(device)

        cam = gradcam(x)
        cam_norm = normalize_map(cam)

        padi = padi_map_for_image(model, x[0], stats)
        padi_norm = normalize_map(padi)
        padi_224 = cv2.resize(padi_norm, (224, 224), interpolation=cv2.INTER_LINEAR)

        sim = cosine_similarity(cam_norm, padi_224)
        iou_vs_padi = iou_at_threshold(cam_norm, padi_224, threshold=0.5)

        mask = np.zeros((224, 224), dtype=np.uint8)
        if path.parent.name != "good":
            gt = DATA_ROOT / "ground_truth" / path.parent.name / (path.stem + "_mask.png")
            mask = load_mask(gt)

        iou_metrics = {}
        if path.parent.name != "good":
            for threshold in (0.3, 0.5, 0.7):
                iou_metrics[f"iou_vs_gt_at_{threshold}"] = iou_at_threshold(cam_norm, mask, threshold=threshold)
        else:
            for threshold in (0.3, 0.5, 0.7):
                iou_metrics[f"iou_vs_gt_at_{threshold}"] = None

        overlay = gradcam.overlay_on_image(img_arr, cam_norm)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_arr)
        axes[0].set_title("Original")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = RESULTS_DIR / f"{path.parent.name}_{path.stem}_gradcam.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        record = {
            "image": str(path.relative_to(PROJECT_ROOT)),
            "defect_type": path.parent.name,
            "cosine_similarity": round(float(sim), 4),
            "iou_vs_padi_at_0.5": round(float(iou_vs_padi), 4),
        }
        for threshold, value in iou_metrics.items():
            record[threshold] = None if value is None else round(float(value), 4)
        agreement.append(record)

    print("Grad-CAM vs PaDiM/ground-truth localization agreement")
    for item in agreement:
        print(item)

    with open(PROJECT_ROOT / "results" / "gradcam_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)


if __name__ == "__main__":
    main()