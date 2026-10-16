"""Sign classification in plain NumPy, so serving the model needs no PyTorch.

`training/train.py` exports the trained PyTorch MLP to `models/sign_mlp.json` + `models/sign_mlp.bin`,
with each BatchNorm folded into the linear layer before it. The browser runs the same files with
`frontend/classifier.js`; the forward pass is just matrix products and GELU activations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FEATURE_VERSION, featurize, pack_hands, to_pixels


def erf(x: np.ndarray) -> np.ndarray:
    """Error function (Abramowitz & Stegun 7.1.26, max error 1.5e-7)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


@dataclass
class Prediction:
    label: str | None  # None when no hand was found
    confidence: float
    top: list[dict] = field(default_factory=list)  # [{"label": "A", "prob": 0.93}, ...]
    num_hands: int = 0

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4), "top": self.top, "num_hands": self.num_hands}


class SignClassifier:
    def __init__(self, layers: Sequence[tuple[np.ndarray, np.ndarray]], classes: Sequence[str], metrics: dict | None = None):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.classes = list(classes)
        self.metrics = metrics or {}

    @classmethod
    def load(cls, path: str | Path) -> "SignClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Sign model not found at {path}. Train it with:\n"
                "  python -m training.extract_landmarks && python -m training.train"
            )
        meta = json.loads(path.read_text())
        if meta.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(f"{path} was exported for feature version {meta.get('feature_version')}, "
                               f"this code uses {FEATURE_VERSION}. Retrain with `python -m training.train`.")
        weights = np.fromfile(path.parent / meta["weights"], dtype="<f4")
        layers = []
        for layer in meta["layers"]:
            n_in, n_out, w0, b0 = layer["in"], layer["out"], layer["weight_offset"], layer["bias_offset"]
            layers.append((weights[w0 : w0 + n_in * n_out].reshape(n_out, n_in), weights[b0 : b0 + n_out]))
        return cls(layers, meta["classes"], meta.get("metrics"))

    def probabilities(self, feats: np.ndarray) -> np.ndarray:
        x = np.asarray(feats, np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = gelu(x)
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def predict_pixels(self, hands_px: Sequence[np.ndarray], top_k: int = 3) -> Prediction:
        """Classify hands given as (21, 3) landmark arrays in pixel coordinates."""
        if not hands_px:
            return Prediction(label=None, confidence=0.0)
        slots, present = pack_hands(hands_px)
        probs = self.probabilities(featurize(slots[None], present[None]))[0]
        order = np.argsort(-probs)[: min(top_k, len(self.classes))]
        top = [{"label": self.classes[i], "prob": round(float(probs[i]), 4)} for i in order]
        return Prediction(label=top[0]["label"], confidence=float(probs[order[0]]), top=top,
                          num_hands=min(len(hands_px), 2))

    def predict_landmarks(self, hands: Sequence[np.ndarray], width: int, height: int, top_k: int = 3) -> Prediction:
        """Classify hands given in MediaPipe's normalised coordinates (what the browser sends)."""
        return self.predict_pixels([to_pixels(h, width, height) for h in hands], top_k)
