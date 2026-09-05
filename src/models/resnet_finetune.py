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


def build_finetune_model(device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.layer4.parameters():
        p.requires_grad = True
    net.fc = nn.Linear(512, 4)
    return net.to(device)


def rotate_batch(x):
    rots = torch.randint(0, 4, (x.size(0),))
    out = torch.stack([torch.rot90(img, k.item(), dims=[1, 2]) for img, k in zip(x, rots)])
    return out, rots


def train(net, loader, device, epochs, lr):
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, _ in loader:
            x_rot, y = rotate_batch(x)
            x_rot, y = x_rot.to(device), y.to(device)
            out = net(x_rot)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        print("epoch %02d | loss %.4f | rotation_acc %.3f" % (epoch, total_loss / n, correct / n))


def extract_embeddings(net, loader, device):
    net.fc = nn.Identity()
    net.eval()
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
    epochs = 20
    lr = 1e-4
    batch_size = 16

    train_ds = MVTecFull(ROOT, "train")
    test_ds = MVTecFull(ROOT, "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("device:", device, "| train:", len(train_ds))

    net = build_finetune_model(device)
    train(net, train_loader, device, epochs, lr)

    torch.save(net.state_dict(), "models/resnet18_finetuned.pt")

    train_emb, train_labels = extract_embeddings(net, eval_train_loader, device)
    test_emb, test_labels = extract_embeddings(net, test_loader, device)

    print("train_emb:", train_emb.shape)
    print("test_emb:", test_emb.shape)

    torch.save({
        "train_emb": train_emb,
        "train_labels": train_labels,
        "train_paths": [str(p) for p in train_ds.paths],
        "test_emb": test_emb,
        "test_labels": test_labels,
        "test_paths": [str(p) for p in test_ds.paths],
    }, "models/resnet18_finetuned_embeddings.pt")


if __name__ == "__main__":
    main()