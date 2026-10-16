# split_images.py
import os, shutil, random
from pathlib import Path

SEED = 42
random.seed(SEED)

# paths
SRC_ROOT = Path("dataset")            # existing dataset root with class folders
DST_ROOT = Path("dataset/test")       # new folder for held-out test set

# parameters
TEST_FRACTION = 0.15                 # 15% of each class → test
ALLOW_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOW_EXT

# discover classes (only folders that actually contain at least 1 image)
classes = []
for d in sorted(SRC_ROOT.iterdir()):
    if d.is_dir():
        imgs = [p for p in d.iterdir() if is_img(p)]
        if imgs:
            classes.append(d.name)

# make output folders
for c in classes:
    (DST_ROOT / c).mkdir(parents=True, exist_ok=True)

moved = {"test": 0}
for c in classes:
    class_dir = SRC_ROOT / c
    imgs = [p for p in class_dir.iterdir() if is_img(p)]
    imgs.sort()
    random.shuffle(imgs)

    n = len(imgs)
    if n == 0:
        print(f"[WARN] No images for class {c}, skipping.")
        continue

    k = max(1, int(TEST_FRACTION * n))  # at least 1
    test_imgs = imgs[:k]

    for src in test_imgs:
        dst = DST_ROOT / c / src.name
        shutil.move(src, dst)
  # copy (safe). Change to move if you want removal.
    moved["test"] += len(test_imgs)

    print(f"[OK] {c:>3}: {k} / {n} copied to TEST")

print("\nClasses:", classes)
print("Total copied to TEST:", moved["test"])
print("Done. Check:", DST_ROOT)
