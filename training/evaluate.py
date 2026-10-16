"""End-to-end evaluation on a folder of labelled photos (one sub-folder per sign).

Use this with photos the model has never seen, ideally of a different person and background:
that is the number that tells you how the app will behave in practice.

    python -m training.evaluate Numbers
    python -m training.evaluate my_photos --model models/sign_mlp.pt --show-errors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signspeak.landmarks import load_image
from signspeak.recognizer import SignRecognizer

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_folder(recognizer: SignRecognizer, folder: Path, verbose: bool = True) -> dict:
    folder = Path(folder)
    rows, skipped = [], set()
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        label = class_dir.name.upper()
        if label not in recognizer.classes:
            skipped.add(class_dir.name)
            continue
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            pred = recognizer.predict_image(load_image(f))
            rows.append({"path": str(f), "true": label, "pred": pred.label, "confidence": round(pred.confidence, 3),
                         "top": [t["label"] for t in pred.top]})

    n = len(rows)
    detected = [r for r in rows if r["pred"] is not None]
    correct = sum(r["pred"] == r["true"] for r in rows)
    top3 = sum(r["true"] in r["top"] for r in rows)
    summary = {
        "images": n,
        "hand_detected": len(detected),
        "accuracy": round(correct / n, 4) if n else None,
        "top3_accuracy": round(top3 / n, 4) if n else None,
        "skipped_classes": sorted(skipped),
    }
    if verbose:
        print(f"\n{folder}: {correct}/{n} correct = {100 * correct / max(n, 1):.1f}%  "
              f"(top-3 {100 * top3 / max(n, 1):.1f}%, hands found in {len(detected)}/{n})")
        if skipped:
            print(f"  skipped folders not in the model's classes: {', '.join(sorted(skipped))}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=Path("models/sign_mlp.pt"))
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--json", type=Path, help="write per-image results here")
    args = ap.parse_args()

    recognizer = SignRecognizer.load(args.model)
    results = {}
    for folder in args.folders:
        res = evaluate_folder(recognizer, folder)
        results[str(folder)] = res
        if args.show_errors:
            for r in res["rows"]:
                if r["pred"] != r["true"]:
                    print(f"  {r['path']}: true {r['true']}  predicted {r['pred']} ({r['confidence']})  top-3 {r['top']}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
