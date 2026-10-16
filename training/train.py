"""Train the landmark classifier on landmarks cached by `training.extract_landmarks`.

    python -m training.train                       # train, evaluate, save models/sign_mlp.pt
    python -m training.train --external Numbers    # also test end-to-end on another labelled image folder

Train/val/test split
--------------------
The ISL dataset images are consecutive video frames, so neighbouring files are near-duplicates.
A random split would put almost the same frame in train and test and report inflated accuracy.
Instead, each class is cut into contiguous blocks of frames (by file number) and whole blocks are
assigned to train / val / test.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from signspeak.features import augment, featurize
from signspeak.model import LandmarkMLP, save_checkpoint


def blocked_split(labels: np.ndarray, paths: np.ndarray, block: int, seed: int, val_frac: float, test_frac: float):
    """Return a train/val/test assignment (0/1/2) per sample, splitting by contiguous frame blocks."""
    rng = random.Random(seed)
    split = np.zeros(len(labels), dtype=np.int64)
    groups = np.empty(len(labels), dtype=object)
    for i, p in enumerate(paths):
        m = re.search(r"(\d+)", Path(p).stem)
        groups[i] = int(m.group(1)) // block if m else hash(Path(p).stem) % 10_000
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        blocks = sorted(set(groups[idx]))
        rng.shuffle(blocks)
        n_test = max(1, round(len(blocks) * test_frac))
        n_val = max(1, round(len(blocks) * val_frac))
        test_b, val_b = set(blocks[:n_test]), set(blocks[n_test : n_test + n_val])
        for i in idx:
            split[i] = 2 if groups[i] in test_b else 1 if groups[i] in val_b else 0
    return split


@torch.no_grad()
def predict(model: nn.Module, feats: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(feats[i : i + batch])).argmax(1).numpy() for i in range(0, len(feats), batch)]
    return np.concatenate(out)


def save_plots(history: list[dict], cm: np.ndarray, classes: list[str], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train")
    a1.set_title("Training loss"), a1.set_xlabel("epoch")
    a2.plot(epochs, [h["train_acc"] for h in history], label="train (augmented)")
    a2.plot(epochs, [h["val_acc"] for h in history], label="validation")
    a2.set_title("Accuracy"), a2.set_xlabel("epoch"), a2.legend()
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True"), ax.set_title("Test-set confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/landmarks.npz"))
    ap.add_argument("--out", type=Path, default=Path("models/sign_mlp.pt"))
    ap.add_argument("--reports", type=Path, default=Path("reports/landmark_model"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--rotation", type=float, default=90.0,
                    help="max random in-plane rotation (degrees); covers tilted hands and sideways phone photos")
    ap.add_argument("--block", type=int, default=50, help="frames per block for the train/val/test split")
    ap.add_argument("--min-samples", type=int, default=200, help="drop classes with fewer usable samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--external", type=Path, nargs="*", default=[], help="labelled image folders for end-to-end testing")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.data, allow_pickle=False)
    hands, present, labels, paths = data["hands"], data["present"], data["labels"], data["paths"]

    counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    dropped = sorted(c for c, n in counts.items() if n < args.min_samples)
    if dropped:
        print(f"Dropping classes with < {args.min_samples} usable samples: "
              + ", ".join(f"{c} ({counts[c]})" for c in dropped))
        keep = ~np.isin(labels, dropped)
        hands, present, labels, paths = hands[keep], present[keep], labels[keep], paths[keep]

    classes = sorted(np.unique(labels).tolist())
    y = np.searchsorted(classes, labels)
    split = blocked_split(labels, paths, args.block, args.seed, val_frac=0.15, test_frac=0.15)
    tr, va, te = (np.flatnonzero(split == k) for k in range(3))
    print(f"{len(classes)} classes | train {len(tr)}  val {len(va)}  test {len(te)}")

    x_val = featurize(hands[va], present[va])
    x_test = featurize(hands[te], present[te])

    model = LandmarkMLP(x_val.shape[1], len(classes), args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        h_aug, p_aug = augment(hands[tr], present[tr], rng, max_rotation_deg=args.rotation)  # fresh augmentation every epoch
        x_tr = torch.from_numpy(featurize(h_aug, p_aug))
        y_tr = torch.from_numpy(y[tr])
        perm = torch.randperm(len(tr))

        model.train()
        loss_sum, correct = 0.0, 0
        for i in range(0, len(tr), args.batch_size):
            b = perm[i : i + args.batch_size]
            logits = model(x_tr[b])
            loss = crit(logits, y_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(b)
            correct += (logits.argmax(1) == y_tr[b]).sum().item()

        val_acc = accuracy_score(y[va], predict(model, x_val))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(tr), "train_acc": correct / len(tr), "val_acc": val_acc})
        print(f"epoch {epoch:3d}  loss {loss_sum / len(tr):.4f}  train {correct / len(tr):.4f}  val {val_acc:.4f}  ({time.time() - t0:.1f}s)")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stop (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    y_pred = predict(model, x_test)
    test_acc = accuracy_score(y[te], y_pred)
    test_f1 = f1_score(y[te], y_pred, average="macro")
    cm = confusion_matrix(y[te], y_pred, labels=range(len(classes)))
    print(f"\nbest epoch {best_epoch}: val acc {best_acc:.4f}")
    print(f"TEST accuracy {test_acc:.4f}   macro-F1 {test_f1:.4f}   ({len(te)} held-out images)")

    metrics = {
        "classes": classes,
        "dropped_classes": dropped,
        "samples": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_epoch": best_epoch,
        "val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "split": f"blocked by {args.block} consecutive frames per class, 70/15/15",
        "hyperparameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "external"},
    }

    if args.external:
        from training.evaluate import evaluate_folder
        from signspeak.recognizer import SignRecognizer

        recognizer = SignRecognizer.from_model(model, classes)
        metrics["external"] = {}
        for folder in args.external:
            result = evaluate_folder(recognizer, folder)
            metrics["external"][str(folder)] = result["summary"]

    save_checkpoint(args.out, model, classes, args.hidden, args.dropout,
                    {k: metrics[k] for k in ("val_accuracy", "test_accuracy", "test_macro_f1")})
    args.reports.mkdir(parents=True, exist_ok=True)
    report = classification_report(y[te], y_pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.reports / "classification_report.json").write_text(json.dumps(report, indent=2))
    (args.reports / "history.json").write_text(json.dumps(history, indent=2))
    save_plots(history, cm, classes, args.reports)
    print(f"Saved model to {args.out} and reports to {args.reports}/")


if __name__ == "__main__":
    main()
