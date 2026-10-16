"""API tests against the real trained model (skipped if the model files are missing)."""
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signspeak.config import PROJECT_ROOT, get_settings
from signspeak.server import create_app

SIGNS_DIR = PROJECT_ROOT / "frontend" / "assets" / "signs"
settings = replace(get_settings(), azure_speech_key=None, azure_speech_region=None, enable_grammar=False)

pytestmark = pytest.mark.skipif(
    not (settings.sign_model_path.exists() and settings.hand_model_path.exists()),
    reason="trained model not available",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(settings)) as c:
        yield c


def post_image(client, path: Path):
    with open(path, "rb") as f:
        return client.post("/api/predict/image", files={"file": (path.name, f, "image/jpeg")})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 35
    assert body["features"] == {"azure_tts": False, "grammar": False}


@pytest.mark.parametrize("label", ["A", "L", "5", "Y", "W"])
def test_predict_reference_images(client, label):
    res = post_image(client, SIGNS_DIR / f"{label}.jpg")
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == label
    assert body["num_hands"] >= 1
    assert len(body["landmarks"][0]) == 21


def test_landmark_endpoint_agrees_with_image_endpoint(client):
    image = post_image(client, SIGNS_DIR / "B.jpg").json()
    res = client.post("/api/predict/landmarks", json={"hands": image["landmarks"], "width": 128, "height": 128})
    assert res.status_code == 200
    assert res.json()["label"] == image["label"] == "B"


def test_no_hands_gives_no_label(client):
    body = client.post("/api/predict/landmarks", json={"hands": [], "width": 640, "height": 480}).json()
    assert body["label"] is None and body["num_hands"] == 0


def test_landmarks_must_have_21_points(client):
    res = client.post("/api/predict/landmarks", json={"hands": [[[0.1, 0.2, 0.0]] * 5], "width": 640, "height": 480})
    assert res.status_code == 422


def test_rejects_non_images(client):
    res = client.post("/api/predict/image", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_speak_without_azure_is_503(client):
    assert client.post("/api/speak", json={"text": "hello"}).status_code == 503


def test_grammar_disabled_echoes_text(client):
    assert client.post("/api/grammar", json={"text": "i am go"}).json() == {"text": "i am go", "corrected": False}


def test_serves_frontend_and_hand_model(client):
    assert "SignSpeak" in client.get("/").text
    assert client.get("/models/hand_landmarker.task").status_code == 200
