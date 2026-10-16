"""Run MediaPipe over an image dataset once and cache the hand landmarks.

Expected layout: one folder per class, named after the sign (e.g. Indian/A/*.jpg, Indian/7/*.jpg).
Folders whose name is not a single letter or digit are skipped.

    python -m training.extract_landmarks --data Indian --out data/landmarks.npz
"""
from __future__ import annotations

import argparse
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from signspeak.features import MAX_HANDS, NUM_LANDMARKS, pack_hands, to_pixels

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_DIR = re.compile(r"^[0-9A-Za-z]$")

_detector = None


def _init_worker(model_path: str) -> None:
    global _detector
    from signspeak.landmarks import HandDetector

    _detector = HandDetector(model_path)


def _process(path: str):
    from signspeak.landmarks import load_image

    try:
        detected = _detector.detect(load_image(path))
    except Exception as exc:  # unreadable / corrupt file
        return path, None, None, 0, str(exc)
    hands_px = [to_pixels(lm, detected.width, detected.height) for lm in detected.landmarks]
    if not hands_px:
        return path, None, None, 0, None
    slots, present = pack_hands(hands_px)
    return path, slots, present, len(hands_px), None


def list_images(root: Path) -> list[tuple[Path, str]]:
    items = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and CLASS_DIR.match(p.name)):
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                items.append((f, class_dir.name.upper()))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("Indian"), help="dataset root with one folder per class")
    ap.add_argument("--out", type=Path, default=Path("data/landmarks.npz"))
    ap.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    items = list_images(args.data)
    if not items:
        raise SystemExit(f"No class folders with images found under {args.data}")
    label_of = {str(p): c for p, c in items}
    print(f"{len(items)} images in {len(set(label_of.values()))} class folders under {args.data}")

    hands, present, labels, paths = [], [], [], []
    missed: dict[str, int] = {}
    errors = 0
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(str(args.model),)) as pool:
        for i, (path, slots, pres, n, err) in enumerate(pool.imap(_process, [str(p) for p, _ in items], chunksize=32), 1):
            label = label_of[path]
            if err:
                errors += 1
            if slots is None:
                missed[label] = missed.get(label, 0) + 1
            else:
                hands.append(slots)
                present.append(pres)
                labels.append(label)
                paths.append(str(Path(path).relative_to(args.data)))
            if i % 2000 == 0 or i == len(items):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(items)}  ({rate:.0f} img/s, {len(labels)} with hands)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        hands=np.asarray(hands, dtype=np.float32).reshape(-1, MAX_HANDS, NUM_LANDMARKS, 3),
        present=np.asarray(present, dtype=bool).reshape(-1, MAX_HANDS),
        labels=np.asarray(labels),
        paths=np.asarray(paths),
    )
    print(f"\nSaved {len(labels)} samples to {args.out} in {time.time() - t0:.0f}s")
    if errors:
        print(f"  {errors} files could not be read")
    for label, n in sorted(missed.items()):
        total = sum(1 for _, c in items if c == label)
        print(f"  no hand detected: {label}: {n}/{total}")


if __name__ == "__main__":
    main()
