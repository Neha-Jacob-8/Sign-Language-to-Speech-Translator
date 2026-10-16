# test.py — plain run, no CLI args
import os, glob, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---------------- Model ----------------
class SmallSignCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        def blk(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2)
            )
        self.f = nn.Sequential(
            blk(3,32), blk(32,64), blk(64,128), blk(128,256),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.h = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256,256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256,n_classes)
        )
    def forward(self,x):
        x = self.f(x)
        x = x.reshape(x.size(0), -1)
        return self.h(x)

# ---------------- Dataset ----------------
class RemappedImageFolder(Dataset):
    def __init__(self, root: str, transform, expected_classes):
        base = datasets.ImageFolder(root=root, transform=None)
        self.loader = base.loader
        self.transform = transform
        exp_map = {c:i for i,c in enumerate(expected_classes)}
        samples = []
        for path, y_old in base.samples:
            cname = base.classes[y_old]
            if cname in exp_map:
                samples.append((path, exp_map[cname]))
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path)
        if self.transform: img = self.transform(img)
        return img, y

# ---------------- Main ----------------
def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1. load checkpoints
    ckpt_paths = sorted(glob.glob("checkpoints/*.pt"))
    if not ckpt_paths:
        raise SystemExit("No checkpoints found in checkpoints/")
    head = torch.load(ckpt_paths[0], map_location="cpu")
    classes = head["classes"]
    n_classes = len(classes)
    print(f"Using {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths: print(" -", p)

    models = []
    for p in ckpt_paths:
        blob = torch.load(p, map_location="cpu")
        m = SmallSignCNN(n_classes)
        m.load_state_dict(blob["model"])
        m.to(device).eval()
        models.append(m)

    # 2. build test dataset
    test_dir = "test"  # augmented hold-out split created by split.py
    tf = transforms.Compose([
        transforms.Resize(int(224*1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    test_ds = RemappedImageFolder(test_dir, tf, classes)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # 3. evaluate
    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for xb,yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = None
            for m in models:
                out = m(xb)
                logits = out if logits is None else (logits + out)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total if total else 0
    print(f"\nTest accuracy: {acc*100:.2f}%   (images: {total})")
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
