import os
import shutil

# ---- Paths ----
BASE = "/Users/nehajacob/Desktop/YourFolder/sem5/proj/deep/new"
SRC_NUMBERS = os.path.join(BASE, "Numbers")
DEST_ROOT = os.path.join(BASE, "one_from_each")  # already created in the previous step

# Acceptable classes and image types
CLASSES = {*(str(d) for d in range(10)), *(chr(c) for c in range(ord("A"), ord("Z")+1))}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def nearest_class_dir_for(path: str, stop_root: str) -> str | None:
    """
    Walk up from 'path' to 'stop_root' and return the first directory name
    that is a valid class (0-9 or A-Z). None if not found.
    """
    path = os.path.abspath(path)
    stop_root = os.path.abspath(stop_root)
    cur = os.path.dirname(path)
    while True:
        name = os.path.basename(cur).strip().upper()
        if name in CLASSES:
            return name
        if os.path.abspath(cur) == stop_root:
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def safe_copy(src: str, dst_dir: str):
    """
    Copy src to dst_dir; avoid overwriting by adding a numeric suffix.
    """
    os.makedirs(dst_dir, exist_ok=True)
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
    if not os.path.isdir(SRC_NUMBERS):
        print(f"[ERROR] Numbers folder not found: {SRC_NUMBERS}")
        return
    if not os.path.isdir(DEST_ROOT):
        print(f"[ERROR] Destination root not found (run the previous step first): {DEST_ROOT}")
        return

    copied = 0
    skipped = 0
    missing_class = 0
    not_image = 0

    for dirpath, _, filenames in os.walk(SRC_NUMBERS):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            if not os.path.isfile(src):
                continue
            if not is_image(src):
                not_image += 1
                continue

            # find nearest parent class folder between file and SRC_NUMBERS
            cls = nearest_class_dir_for(src, SRC_NUMBERS)
            if cls is None:
                # No valid class name found in path — skip
                missing_class += 1
                continue

            dst_cls_dir = os.path.join(DEST_ROOT, cls)
            _ = safe_copy(src, dst_cls_dir)
            copied += 1

            if copied % 200 == 0:
                print(f"[INFO] Copied so far: {copied}…")

    print("\n[SUMMARY]")
    print(f"  Copied images: {copied}")
    print(f"  Skipped non-images: {not_image}")
    print(f"  Skipped (no class folder in path): {missing_class}")
    print(f"  Destination: {DEST_ROOT}")

if __name__ == "__main__":
    main()
