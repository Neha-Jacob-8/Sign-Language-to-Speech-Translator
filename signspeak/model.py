"""The sign classifier: a small fully-connected network over hand-landmark features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
