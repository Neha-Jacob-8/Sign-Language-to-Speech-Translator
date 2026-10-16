"""The browser classifier (frontend/classifier.js) must agree with the Python one."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from signspeak.classifier import SignClassifier
from signspeak.config import PROJECT_ROOT
from signspeak.features import featurize, pack_hands, to_pixels

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "landmarks.json").read_text())
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")


@pytest.fixture(scope="module")
def js_results():
    script = Path(__file__).parent / "js" / "classify_fixtures.mjs"
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
    return json.loads(out.stdout)


def python_features():
    rows = []
    for s in FIXTURES:
        slots, present = pack_hands([to_pixels(np.array(h), s["width"], s["height"]) for h in s["hands"]])
        rows.append(featurize(slots[None], present[None])[0])
    return np.stack(rows)


def test_features_match(js_results):
    np.testing.assert_allclose([r["features"] for r in js_results], python_features(), atol=1e-4)


def test_probabilities_match(js_results):
    expected = SignClassifier.load(PROJECT_ROOT / "models" / "sign_mlp.json").probabilities(python_features())
    np.testing.assert_allclose([r["probabilities"] for r in js_results], expected, atol=1e-5)


def test_browser_labels(js_results):
    assert [r["label"] for r in js_results] == [s["label"] for s in FIXTURES]
