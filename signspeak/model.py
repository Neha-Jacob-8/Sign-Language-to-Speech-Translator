"""The sign classifier: a small fully-connected network over hand-landmark features (training side)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_DIM, FEATURE_VERSION


class LandmarkMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: Sequence[int] = (256, 256, 128), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(path: str | Path, model: LandmarkMLP, classes: Sequence[str], hidden: Sequence[int],
                    dropout: float, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "in_dim": FEATURE_DIM,
            "hidden": list(hidden),
            "dropout": dropout,
            "feature_version": FEATURE_VERSION,
            "metrics": metrics,
        },
        path,
    )


def export_inference(model: LandmarkMLP, classes: Sequence[str], path: str | Path, metrics: dict[str, Any]) -> None:
    """Write the model for inference without PyTorch: `<name>.json` (metadata) + `<name>.bin` (float32 weights).

    The same files are loaded by `signspeak.classifier` in Python and by `frontend/classifier.js`
    in the browser. Each Linear -> BatchNorm pair is folded into one linear layer (exact in eval mode):
    y = gamma * (Wx + b - mean) / sqrt(var + eps) + beta
    """
    model = model.eval()
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    norms = [m for m in model.net if isinstance(m, nn.BatchNorm1d)]
    blobs, layers, offset = [], [], 0
    with torch.no_grad():
        for i, lin in enumerate(linears):
            w, b = lin.weight.clone(), lin.bias.clone()
            if i < len(norms):
                bn = norms[i]
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                w = w * scale[:, None]
                b = (b - bn.running_mean) * scale + bn.bias
            w = w.numpy().astype("<f4")
            b = b.numpy().astype("<f4")
            layers.append({"in": int(w.shape[1]), "out": int(w.shape[0]), "weight_offset": offset, "bias_offset": offset + w.size})
            offset += w.size + b.size
            blobs += [w.ravel(), b]
    path = Path(path).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(blobs).astype("<f4").tofile(path.with_suffix(".bin"))
    path.write_text(json.dumps({
        "format": "signspeak-mlp",
        "feature_version": FEATURE_VERSION,
        "activation": "gelu",
        "classes": list(classes),
        "metrics": metrics,
        "weights": path.with_suffix(".bin").name,
        "layers": layers,
    }, indent=1))


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[LandmarkMLP, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"{path} was trained with feature version {ckpt.get('feature_version')}, "
            f"but this code produces version {FEATURE_VERSION}. Retrain with `python -m training.train`."
        )
    model = LandmarkMLP(ckpt["in_dim"], len(ckpt["classes"]), ckpt["hidden"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
