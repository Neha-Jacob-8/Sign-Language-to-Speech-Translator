"""Hand landmark detection with MediaPipe Hand Landmarker (used for uploaded images and training data).

In live camera mode the browser runs the same MediaPipe model itself and only sends the
landmarks to the server, so this module is not on the real-time path.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ info logs

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

# Very large photos (phones) are downscaled first: landmarks are normalised, so this only
# saves time, and MediaPipe internally works at a much lower resolution anyway.
MAX_SIDE = 1280


@dataclass
class DetectedHands:
    """Landmarks of every detected hand, in MediaPipe's normalised [0, 1] image coordinates."""

    landmarks: list[np.ndarray]  # each (21, 3)
    handedness: list[str]
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.landmarks)


def load_image(source) -> Image.Image:
    """Open an image (path or file-like), apply EXIF rotation and convert to RGB."""
    image = Image.open(source)
    image = ImageOps.exif_transpose(image)  # phone photos are often stored sideways
    return image.convert("RGB")


class HandDetector:
    def __init__(self, model_path: str | Path, max_hands: int = 2, min_confidence: float = 0.3):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}. Download it with:\n"
                "  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # the landmarker is not safe to call from several threads

    def detect(self, image: Image.Image) -> DetectedHands:
        if max(image.size) > MAX_SIDE:
            image = image.copy()
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.BILINEAR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(image)))
        with self._lock:
            result = self._landmarker.detect(mp_image)
        landmarks = [np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32) for hand in result.hand_landmarks]
        handedness = [cats[0].category_name if cats else "" for cats in result.handedness]
        return DetectedHands(landmarks, handedness, image.width, image.height)

    def close(self) -> None:
        self._landmarker.close()
