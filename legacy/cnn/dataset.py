import os, time, random, json, hashlib
from io import BytesIO
from collections import defaultdict
from typing import Dict, List

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# ---------- CONFIG ----------
BASE = "/Users/nehajacob/Desktop/YourFolder/sem5/proj/deep/new"

SEED_ROOT = os.path.join(BASE, "one_from_each")   # your small seed set
OUT_ROOT  = os.path.join(BASE, "dataset")         # single folder (no split)

TOTAL = 42744            # total required across all classes
IMG_SIZE = 256           # final size
OUT_EXT = ".png"         # use PNG for stable hashing

CACHE = os.path.join(BASE, ".uniq_hashes_dataset.json")  # resume cache of saved PNG hashes

CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z")+1)]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

RANDOM_SEED = 12345
# ----------------------------

random.seed(RANDOM_SEED)


def ensure_dirs():
    for c in CLASSES:
        os.makedirs(os.path.join(OUT_ROOT, c), exist_ok=True)

def list_seed() -> Dict[str, List[str]]:
    per_class = defaultdict(list)
    for c in CLASSES:
        p = os.path.join(SEED_ROOT, c)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.lower().endswith(IMAGE_EXTS):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    per_class[c].append(fp)
    return per_class

def bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_hash_cache() -> Dict[str, bool]:
    h = {}
    if os.path.isfile(CACHE):
        try:
            with open(CACHE, "r") as f:
                h = json.load(f)
        except Exception:
            pass
    # also index anything already in OUT_ROOT so we can resume
    for c in CLASSES:
        d = os.path.join(OUT_ROOT, c)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(".png"):
                fp = os.path.join(d, name)
                try:
                    hh = file_hash(fp)
                    h[hh] = True
                except Exception:
                    pass
    return h

def save_cache(h: Dict[str, bool]):
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(h, f)
    os.replace(tmp, CACHE)

def augment_once(im: Image.Image, rng: random.Random) -> Image.Image:
    im = im.convert("RGB")

    # pad/crop jitter to square canvas
    target_side = max(96, min(512, int(max(im.size) * rng.uniform(0.8, 1.2))))
    bg = (rng.randint(0, 30),) * 3
    im = ImageOps.pad(im, (target_side, target_side), color=bg, centering=(0.5, 0.5))

    # rotate ±25°
    angle = rng.uniform(-25, 25)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=False)

    # shear x (approx ±20°)
    shear = rng.uniform(-0.35, 0.35)
    m = (1, shear, 0, 0, 1, 0)
    im = im.transform(im.size, Image.AFFINE, m, resample=Image.BICUBIC)

    # scale jitter
    if rng.random() < 0.7:
        s = rng.uniform(0.85, 1.15)
        im = im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))), Image.BICUBIC)
        im = ImageOps.fit(im, (target_side, target_side), Image.BICUBIC)

    # color jitter
    if rng.random() < 0.9: im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.9: im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.4))
    if rng.random() < 0.5: im = ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.5))

    # blur or sharpen or none
    p = rng.random()
    if p < 0.33:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.0)))
    elif p < 0.66:
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # gaussian noise
    if rng.random() < 0.8:
        arr = np.array(im).astype(np.int16)
        sigma = rng.uniform(3, 12)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, "RGB")

    # random cutout
    if rng.random() < 0.4:
        arr = np.array(im)
        h, w, _ = arr.shape
        for _ in range(rng.randint(1, 3)):
            cw = rng.randint(w // 12, w // 5)
            ch = rng.randint(h // 12, h // 5)
            x0 = rng.randint(0, max(0, w - cw))
            y0 = rng.randint(0, max(0, h - ch))
            arr[y0:y0+ch, x0:x0+cw, :] = rng.integers(0, 255, size=(ch, cw, 3), dtype=np.uint8)
        im = Image.fromarray(arr, "RGB")

    # grayscale mix
    if rng.random() < 0.4:
        g = ImageOps.grayscale(im).convert("RGB")
        alpha = rng.uniform(0.2, 0.8)
        im = Image.blend(im, g, alpha)

    # finalize size
    im = ImageOps.fit(im, (IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return im

def save_png_and_hash(im: Image.Image):
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=6)
    data = buf.getvalue()
    return data, bytes_hash(data)

def balanced_quotas(total: int) -> Dict[str, int]:
    n = len(CLASSES)  # 36
    base = total // n        # 1187
    rem  = total % n         # 12
    q = {c: base for c in CLASSES}
    for c in CLASSES[:rem]:
        q[c] += 1
    return q

def main():
    ensure_dirs()
    seeds = list_seed()
    if not any(seeds.values()):
        print("[ERROR] No seed images found under:", SEED_ROOT)
        return

    quotas = balanced_quotas(TOTAL)
    # resume support
    saved = load_hash_cache()

    # count already present per class
    existing = {c: 0 for c in CLASSES}
    for c in CLASSES:
        d = os.path.join(OUT_ROOT, c)
        if os.path.isdir(d):
            existing[c] = len([f for f in os.listdir(d) if f.lower().endswith(".png")])

    goal_total = sum(quotas.values())
    have_total = sum(existing.values())
    print(f"[INFO] Goal={goal_total}, already have={have_total}. Generating the rest…")

    for c in CLASSES:
        seed_list = seeds.get(c, [])
        if not seed_list:
            print(f"[WARN] No seed for class {c}; skipping.")
            continue

        need = max(0, quotas[c] - existing[c])
        if need == 0:
            print(f"[OK] {c}: already has {existing[c]}/{quotas[c]}")
            continue

        print(f"[DO] {c}: need {need} more (seeds={len(seed_list)})")
        idx = 0
        made = 0
        stalls = 0

        while need > 0:
            base_path = seed_list[idx % len(seed_list)]
            idx += 1
            try:
                with Image.open(base_path) as im:
                    im = im.convert("RGB")
                    rng = random.Random(hash(base_path) ^ random.getrandbits(64) ^ int(time.time()*1e6))
                    aug = augment_once(im, rng)
                    data, h = save_png_and_hash(aug)

                    if h in saved:
                        stalls += 1
                        if stalls % 500 == 0:
                            print(f"  {c}: still searching unique… (stalls={stalls})")
                        continue

                    # unique — write it
                    out_dir = os.path.join(OUT_ROOT, c)
                    ts = int(time.time() * 1000)
                    name = f"{c}_{ts}_{h[:8]}{OUT_EXT}"
                    out_path = os.path.join(out_dir, name)
                    with open(out_path, "wb") as f:
                        f.write(data)

                    saved[h] = True
                    need -= 1
                    made += 1
                    if made % 100 == 0:
                        print(f"  {c}: generated {made}")
                        save_cache(saved)

            except KeyboardInterrupt:
                print("\n[INFO] Interrupted — saving cache and exiting.")
                save_cache(saved)
                return
            except Exception as e:
                print(f"[WARN] {c}: problem with {base_path}: {e}")
                continue

        save_cache(saved)
        print(f"[DONE] {c}: generated {made}, now at {quotas[c]} total")

    print("\n[SUMMARY]")
    total = 0
    for c in CLASSES:
        d = os.path.join(OUT_ROOT, c)
        n = len([f for f in os.listdir(d) if f.lower().endswith('.png')]) if os.path.isdir(d) else 0
        total += n
        print(f"  {c}: {n}")
    print(f"[OK] Dataset ready at: {OUT_ROOT}  (total={total})")

if __name__ == "__main__":
    main()
