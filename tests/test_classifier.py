import json
import math
from pathlib import Path

import numpy as np
import pytest

from signspeak.classifier import SignClassifier, erf
from signspeak.config import PROJECT_ROOT
from signspeak.features import featurize, pack_hands, to_pixels

MODEL = PROJECT_ROOT / "models" / "sign_mlp.json"
PT = PROJECT_ROOT / "models" / "sign_mlp.pt"
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "landmarks.json").read_text())


def fixture_features():
    rows = []
    for s in FIXTURES:
        slots, present = pack_hands([to_pixels(np.array(h), s["width"], s["height"]) for h in s["hands"]])
        rows.append(featurize(slots[None], present[None])[0])
    return np.stack(rows)


def test_erf_matches_math_erf():
    z = np.linspace(-6, 6, 2001)
    assert np.abs(erf(z) - np.vectorize(math.erf)(z)).max() < 2e-7


def test_probabilities_are_a_distribution():
    probs = SignClassifier.load(MODEL).probabilities(fixture_features())
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_exported_model_matches_pytorch():
    torch = pytest.importorskip("torch")
    from signspeak.model import load_checkpoint

    model, _ = load_checkpoint(PT)
    x = fixture_features()
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    np.testing.assert_allclose(SignClassifier.load(MODEL).probabilities(x), expected, atol=1e-5)
