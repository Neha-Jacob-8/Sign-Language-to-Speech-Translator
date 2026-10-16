import os
import shutil
import random

# --- paths (change only if you want a different dest name) ---
BASE = "/Users/nehajacob/Desktop/YourFolder/sem5/proj/deep/new"
SRC_INDIAN = os.path.join(BASE, "Indian")
DEST_ROOT = os.path.join(BASE, "one_from_each")  # new folder that will be created

# pick one image per class: set to True for random pick, False for first alphabetically
PICK_RANDOM = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def ensure_dest_structure():
    os.makedirs(DEST_ROOT, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, c), exist_ok=True)

def find_class_dir(src_root: str, target: str):
    """Return the path of the class folder in Indian, case-insensitive, only direct child."""
    for name in os.listdir(src_root):
        p = os.path.join(src_root, name)
        if os.path.isdir(p) and name.strip().upper() == target:
            return p
    return None

def pick_one_image(src_dir: str):
    """Find one image in the class folder (recursively)."""
    files = []
    for dp, _, fns in os.walk(src_dir):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if os.path.isfile(fp) and is_image(fp):
                files.append(fp)
    if not files:
        return None
    return random.choice(files) if PICK_RANDOM else sorted(files)[0]

def safe_copy(src: str, dst_dir: str):
    """Copy src to dst_dir; if filename exists, add a suffix."""
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    candidate = os.path.join(dst_dir, base)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dst_dir, f"{stem}__{i}{ext}")
        i += 1
    shutil.copy2(src, candidate)
    return candidate

def main():
    if not os.path.isdir(SRC_INDIAN):
        print(f"[ERROR] Source folder not found: {SRC_INDIAN}")
        return

    ensure_dest_structure()

    copied = 0
    missing = []
    for cls in CLASSES:
        src_cls = find_class_dir(SRC_INDIAN, cls)
        if not src_cls:
            print(f"[WARN] Class folder '{cls}' not found directly under Indian/")
            missing.append(cls)
            continue

        img = pick_one_image(src_cls)
        if not img:
            print(f"[WARN] No images found in: {src_cls}")
            missing.append(cls)
            continue

        dst_cls_dir = os.path.join(DEST_ROOT, cls)
        out = safe_copy(img, dst_cls_dir)
        copied += 1
        print(f"[OK] {cls}: copied '{os.path.basename(img)}' -> '{out}'")

    print("\n[SUMMARY]")
    print(f"  Destination: {DEST_ROOT}")
    print(f"  Copied classes: {copied}/{len(CLASSES)}")
    if missing:
        print(f"  Missing/no-image classes: {', '.join(missing)}")

if __name__ == "__main__":
    main()
