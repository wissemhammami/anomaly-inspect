import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.evaluation.gradcam_eval import MODEL_PATH, load_model
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
ROOT = str(MODEL_PATH.parents[1] / "data" / "bottle")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def load_mask(path):
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return (np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


class TrainGood(Dataset):
    def __init__(self, root):
        self.paths = sorted((Path(root) / "train" / "good").glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image(self.paths[idx])


class TestWithMasks(Dataset):
    def __init__(self, root):
        self.samples = []
        test_dir = Path(root) / "test"
        gt_dir = Path(root) / "ground_truth"
        for d in sorted(test_dir.iterdir()):
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.png")):
                mask_path = gt_dir / d.name / (p.stem + "_mask.png") if d.name != "good" else None
                self.samples.append((p, mask_path, d.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, mask_path, defect_type = self.samples[idx]
        image = load_image(path)
        if mask_path is not None and mask_path.exists():
            mask = load_mask(mask_path)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        return image, mask, defect_type


backbone = load_model(DEVICE)

feat_store = {}


def hook(module, inp, out):
    feat_store["layer2"] = out


backbone.layer2.register_forward_hook(hook)


def get_feature_map(x):
    with torch.no_grad():
        backbone(x)
    return feat_store["layer2"]  # (B, C, H, W)


train_ds = TrainGood(ROOT)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

feats = []
for x in train_loader:
    fm = get_feature_map(x.to(DEVICE))  # (B, C, H, W)
    feats.append(fm.cpu())
feats = torch.cat(feats)  # (N, C, H, W)
N, C, H, W = feats.shape
feats = feats.permute(0, 2, 3, 1).reshape(N, H * W, C)  # (N, HW, C)

mean = feats.mean(dim=0).numpy()  # (HW, C)
cov_inv = np.zeros((H * W, C, C), dtype=np.float32)
for pos in range(H * W):
    cov = np.cov(feats[:, pos, :].numpy(), rowvar=False)
    cov += np.eye(C) * 1e-4
    cov_inv[pos] = np.linalg.inv(cov)

print(f"fitted per-position gaussian: {H}x{W} positions, {C} channels")

test_ds = TestWithMasks(ROOT)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

all_pixel_scores, all_pixel_labels = [], []

for x, mask, defect_type in test_loader:
    fm = get_feature_map(x.to(DEVICE))[0]  # (C, H, W)
    fm = fm.permute(1, 2, 0).reshape(H * W, C).cpu().numpy()

    scores = np.zeros(H * W, dtype=np.float32)
    for pos in range(H * W):
        diff = fm[pos] - mean[pos]
        scores[pos] = diff @ cov_inv[pos] @ diff

    score_map = scores.reshape(1, 1, H, W)
    score_map = F.interpolate(torch.from_numpy(score_map), size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False)
    score_map = score_map.squeeze().numpy()

    all_pixel_scores.append(score_map.flatten())
    all_pixel_labels.append(mask.squeeze().numpy().flatten())

all_pixel_scores = np.concatenate(all_pixel_scores)
all_pixel_labels = np.concatenate(all_pixel_labels)

pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
pixel_pr_auc = average_precision_score(all_pixel_labels, all_pixel_scores)
print(f"pixel-level ROC-AUC: {pixel_auroc:.4f}")
print(f"pixel-level PR-AUC:  {pixel_pr_auc:.4f}")

torch.save({
    "mean": mean.astype(np.float16),
    "cov_inv": cov_inv.astype(np.float16),
    "H": H, "W": W, "C": C,
    "pixel_auroc": pixel_auroc,
    "pixel_pr_auc": pixel_pr_auc,
}, MODEL_PATH.parent / "localization_stats.pt")