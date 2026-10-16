"""API tests against the real trained model, using landmarks saved from the reference sign images."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signspeak.config import get_settings
from signspeak.server import create_app

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "landmarks.json").read_text())
settings = replace(get_settings(), azure_speech_key=None, azure_speech_region=None, enable_grammar=False)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 35
    assert body["metrics"]["test_accuracy"] > 0.99
    assert body["features"] == {"azure_tts": False, "grammar": False}


@pytest.mark.parametrize("sample", FIXTURES, ids=[s["label"] for s in FIXTURES])
def test_predicts_reference_signs(client, sample):
    res = client.post("/api/predict/landmarks", json={k: sample[k] for k in ("hands", "width", "height")})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == sample["label"]
    assert body["num_hands"] == len(sample["hands"])
    assert len(body["top"]) == 3


def test_no_hands_gives_no_label(client):
    body = client.post("/api/predict/landmarks", json={"hands": [], "width": 640, "height": 480}).json()
    assert body["label"] is None and body["num_hands"] == 0


@pytest.mark.parametrize("payload", [
    {"hands": [[[0.1, 0.2, 0.0]] * 5], "width": 640, "height": 480},  # wrong number of points
    {"hands": [[[0.1, 0.2]] * 21], "width": 640, "height": 480},  # missing z
    {"hands": [[[0.1, 0.2, 0.0]] * 21] * 3, "width": 640, "height": 480},  # three hands
    {"hands": [], "width": 0, "height": 480},  # invalid size
])
def test_rejects_malformed_landmarks(client, payload):
    assert client.post("/api/predict/landmarks", json=payload).status_code == 422


def test_speak_without_azure_is_503(client):
    assert client.post("/api/speak", json={"text": "hello"}).status_code == 503


def test_grammar_disabled_echoes_text(client):
    assert client.post("/api/grammar", json={"text": "i am go"}).json() == {"text": "i am go", "corrected": False}


def test_serves_frontend_and_hand_model(client):
    assert "SignSpeak" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    assert client.get("/classifier.js").status_code == 200
    for name in ("hand_landmarker.task", "sign_mlp.json", "sign_mlp.bin"):
        assert client.get(f"/models/{name}").status_code == 200
    assert client.get("/models/sign_mlp.pt").status_code == 404  # only the files the browser needs
