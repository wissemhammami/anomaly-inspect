import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
ROOT = os.environ.get("MVTEC_DATA_ROOT", "data/bottle")


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


class MVTecFull(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self.paths = []
        self.labels = []
        if split == "train":
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                self.paths.append(p)
                self.labels.append(0)
        else:
            for folder in sorted((self.root / "test").iterdir()):
                if not folder.is_dir():
                    continue
                label = 0 if folder.name == "good" else 1
                for p in sorted(folder.glob("*.png")):
                    self.paths.append(p)
                    self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image(self.paths[idx]), self.labels[idx]


def build_backbone(device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device)


def extract_embeddings(net, loader, device):
    embs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = net(x)
            embs.append(out.cpu())
            labels.append(y)
    return torch.cat(embs), torch.cat(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MVTecFull(ROOT, "train")
    test_ds = MVTecFull(ROOT, "test")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    net = build_backbone(device)

    train_emb, train_labels = extract_embeddings(net, train_loader, device)
    test_emb, test_labels = extract_embeddings(net, test_loader, device)

    print("device:", device)
    print("train_emb:", train_emb.shape)
    print("test_emb:", test_emb.shape)

    torch.save({
        "train_emb": train_emb,
        "train_labels": train_labels,
        "train_paths": [str(p) for p in train_ds.paths],
        "test_emb": test_emb,
        "test_labels": test_labels,
        "test_paths": [str(p) for p in test_ds.paths],
    }, "models/resnet18_frozen_embeddings.pt")


if __name__ == "__main__":
    main()