import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
ROOT = "data/bottle"


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


class MVTecTrainOnly(Dataset):
    def __init__(self, root):
        self.paths = sorted((Path(root) / "train" / "good").glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image(self.paths[idx])


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def rotate_batch(x):
    rots = torch.randint(0, 4, (x.size(0),))
    out = torch.stack([torch.rot90(img, k.item(), dims=[1, 2]) for img, k in zip(x, rots)])
    return out, rots


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    lr = 1e-3
    batch_size = 16

    train_ds = MVTecTrainOnly(ROOT)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print("device:", device, "| train:", len(train_ds))

    model = CNNBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x in train_loader:
            x_rot, y = rotate_batch(x)
            x_rot, y = x_rot.to(device), y.to(device)
            out = model(x_rot)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        print("epoch %02d | loss %.4f | rotation_acc %.3f" % (epoch, total_loss / n, correct / n))

    torch.save(model.state_dict(), "models/cnn_baseline.pt")


if __name__ == "__main__":
    main()