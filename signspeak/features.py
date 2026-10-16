"""Turn MediaPipe hand landmarks into a fixed-length, position- and scale-invariant feature vector.

This is the single source of truth for featurisation: training, the API and the tests all use it,
so a model can never see features computed differently from the ones it was trained on.

Layout of one feature vector (FEATURE_DIM = 131):
    [  0:63 ]  hand slot 0 (left-most hand in the image): 21 landmarks x (x, y, z)
    [ 63:126]  hand slot 1 (right-most hand), zeros if only one hand is visible
    [126:128]  presence flags for slot 0 and slot 1
    [128:131]  vector from wrist 0 to wrist 1 (two-handed signs only)

Each hand is expressed relative to its own wrist and divided by its palm length
(wrist -> middle-finger knuckle), so it does not matter where the hand is in the frame,
how far it is from the camera, or what the image resolution is.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

NUM_LANDMARKS = 21
MAX_HANDS = 2
WRIST, MIDDLE_MCP = 0, 9
FEATURE_VERSION = 1
FEATURE_DIM = MAX_HANDS * NUM_LANDMARKS * 3 + MAX_HANDS + 3


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe returns x/y normalised by width/height; undo that so non-square frames keep their shape.

    z uses the same scale as x (as documented by MediaPipe).
    """
    out = np.asarray(landmarks, dtype=np.float32).copy()
    out[..., 0] *= width
    out[..., 1] *= height
    out[..., 2] *= width
    return out


def pack_hands(hands: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Place up to two (21, 3) hands into fixed slots, ordered left-to-right by wrist position."""
    slots = np.zeros((MAX_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)
    present = np.zeros(MAX_HANDS, dtype=bool)
    ordered = sorted((np.asarray(h, dtype=np.float32) for h in hands), key=lambda h: float(h[WRIST, 0]))
    for i, hand in enumerate(ordered[:MAX_HANDS]):
        if hand.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"each hand must have shape (21, 3), got {hand.shape}")
        slots[i] = hand
        present[i] = True
    return slots, present


def order_slots(hands: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-establish the slot invariant for a batch: slot 0 is filled first and is the left-most hand."""
    hands = hands.copy()
    present = present.copy()
    only_second = ~present[:, 0] & present[:, 1]
    swap = (present.all(axis=1) & (hands[:, 0, WRIST, 0] > hands[:, 1, WRIST, 0])) | only_second
    hands[swap] = hands[swap][:, ::-1]
    present[swap] = present[swap][:, ::-1]
    return hands, present


def featurize(hands: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(B, 2, 21, 3) pixel landmarks + (B, 2) presence mask -> (B, FEATURE_DIM) float32 features."""
    hands = np.asarray(hands, dtype=np.float32)
    present = np.asarray(present, dtype=bool)
    batch = hands.shape[0]

    rel = hands - hands[:, :, WRIST : WRIST + 1, :]
    palm = np.linalg.norm(rel[:, :, MIDDLE_MCP, :], axis=-1)  # (B, 2)
    palm = np.maximum(palm, 1e-6)
    shape = rel / palm[:, :, None, None]
    shape *= present[:, :, None, None]

    both = present.all(axis=1)
    mean_palm = np.where(both, palm.mean(axis=1), 1.0)
    between = (hands[:, 1, WRIST, :] - hands[:, 0, WRIST, :]) / mean_palm[:, None]
    between = np.where(both[:, None], between, 0.0)

    feats = np.concatenate(
        [shape.reshape(batch, -1), present.astype(np.float32), between.astype(np.float32)], axis=1
    )
    return np.clip(feats, -20.0, 20.0).astype(np.float32)


def augment(
    hands: np.ndarray,
    present: np.ndarray,
    rng: np.random.Generator,
    mirror_prob: float = 0.5,
    max_rotation_deg: float = 90.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    jitter: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Random geometric augmentation applied to raw pixel landmarks (training only).

    - mirror: makes the model work for left- and right-handed signers and mirrored (selfie) cameras
    - in-plane rotation and anisotropic scaling: camera angle / tilt
    - jitter: landmark noise, scaled to the size of the hand
    """
    hands = np.asarray(hands, dtype=np.float32).copy()
    batch = hands.shape[0]

    flip = rng.random(batch) < mirror_prob
    hands[flip, ..., 0] *= -1.0

    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg, batch)).astype(np.float32)
    cos, sin = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    x, y = hands[..., 0].copy(), hands[..., 1].copy()
    hands[..., 0] = cos * x - sin * y
    hands[..., 1] = sin * x + cos * y

    scale = rng.uniform(*scale_range, size=(batch, 1, 1, 2)).astype(np.float32)
    hands[..., :2] *= scale

    palm = np.linalg.norm(hands[:, :, MIDDLE_MCP] - hands[:, :, WRIST], axis=-1)  # (B, 2)
    palm = (palm * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1)
    noise = rng.normal(0.0, 1.0, hands.shape).astype(np.float32) * (jitter * palm)[:, None, None, None]
    hands += noise * present[:, :, None, None]

    return order_slots(hands, present)
