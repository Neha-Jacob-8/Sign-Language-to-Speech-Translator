"""End-to-end recogniser: hand landmarks -> features -> classifier -> top-k signs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .config import PROJECT_ROOT
from .features import featurize, pack_hands, to_pixels
from .landmarks import HandDetector
from .model import LandmarkMLP, load_checkpoint

DEFAULT_HAND_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"


@dataclass
class Prediction:
    label: str | None  # None when no hand was found
    confidence: float
    top: list[dict] = field(default_factory=list)  # [{"label": "A", "prob": 0.93}, ...]
    num_hands: int = 0
    landmarks: list[list[list[float]]] = field(default_factory=list)  # normalised, for drawing overlays

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top": self.top,
            "num_hands": self.num_hands,
            "landmarks": self.landmarks,
        }


class SignRecognizer:
    def __init__(self, model: LandmarkMLP, classes: Sequence[str], hand_model_path: Path = DEFAULT_HAND_MODEL,
                 metrics: dict | None = None):
        self.model = model.eval()
        self.classes = list(classes)
        self.metrics = metrics or {}
        self._hand_model_path = hand_model_path
        self._detector: HandDetector | None = None

    @classmethod
    def load(cls, sign_model_path: Path, hand_model_path: Path = DEFAULT_HAND_MODEL) -> "SignRecognizer":
        if not Path(sign_model_path).exists():
            raise FileNotFoundError(
                f"Sign model not found at {sign_model_path}. Train it with:\n"
                "  python -m training.extract_landmarks && python -m training.train"
            )
        model, ckpt = load_checkpoint(sign_model_path)
        return cls(model, ckpt["classes"], hand_model_path, ckpt.get("metrics"))

    @classmethod
    def from_model(cls, model: LandmarkMLP, classes: Sequence[str], hand_model_path: Path = DEFAULT_HAND_MODEL):
        return cls(model, classes, hand_model_path)

    @property
    def detector(self) -> HandDetector:
        if self._detector is None:  # created lazily: live mode never needs it
            self._detector = HandDetector(self._hand_model_path)
        return self._detector

    @torch.no_grad()
    def predict_pixels(self, hands_px: Sequence[np.ndarray], top_k: int = 3) -> Prediction:
        """Classify hands given as (21, 3) landmark arrays in pixel coordinates."""
        if not hands_px:
            return Prediction(label=None, confidence=0.0)
        slots, present = pack_hands(hands_px)
        x = torch.from_numpy(featurize(slots[None], present[None]))
        probs = torch.softmax(self.model(x), dim=1)[0]
        values, indices = probs.topk(min(top_k, len(self.classes)))
        top = [{"label": self.classes[i], "prob": round(float(v), 4)} for v, i in zip(values, indices)]
        return Prediction(label=top[0]["label"], confidence=float(values[0]), top=top, num_hands=min(len(hands_px), 2))

    def predict_landmarks(self, hands: Sequence[np.ndarray], width: int, height: int, top_k: int = 3) -> Prediction:
        """Classify hands given in MediaPipe's normalised coordinates (what the browser sends)."""
        return self.predict_pixels([to_pixels(h, width, height) for h in hands], top_k)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def predict_image(self, image: Image.Image, top_k: int = 3) -> Prediction:
        detected = self.detector.detect(image)
        pred = self.predict_landmarks(detected.landmarks, detected.width, detected.height, top_k)
        pred.landmarks = [np.round(lm.astype(np.float64), 4).tolist() for lm in detected.landmarks]
        return pred
