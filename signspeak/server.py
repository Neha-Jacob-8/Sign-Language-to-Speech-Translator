"""Optional FastAPI server: serves the web app locally and adds what a static host cannot,
the Azure neural voice and T5 grammar correction. It also exposes the classifier as an API.
Recognition itself runs in the browser, so the server needs no PyTorch, MediaPipe or OpenCV.

    uvicorn signspeak.server:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__, tts
from .classifier import SignClassifier
from .config import Settings, get_settings
from .features import NUM_LANDMARKS
from .grammar import GrammarCorrector

log = logging.getLogger("signspeak")


class LandmarkRequest(BaseModel):
    """Hands as returned by MediaPipe in the browser: normalised [x, y, z] per landmark."""

    hands: list[list[list[float]]] = Field(..., max_length=2)
    width: int = Field(..., gt=0, le=10_000)
    height: int = Field(..., gt=0, le=10_000)
    top_k: int = Field(3, ge=1, le=10)

    @field_validator("hands")
    @classmethod
    def _check_shape(cls, hands):
        for hand in hands:
            if len(hand) != NUM_LANDMARKS or any(len(p) != 3 for p in hand):
                raise ValueError(f"each hand must be {NUM_LANDMARKS} points of [x, y, z]")
        return hands


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = SignClassifier.load(settings.sign_model_path)
        app.state.grammar = GrammarCorrector() if settings.enable_grammar else None
        log.info("Loaded sign model with %d classes", len(app.state.classifier.classes))
        yield

    app = FastAPI(title="SignSpeak: ISL to Speech", version=__version__, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/api/health")
    def health(request: Request):
        clf: SignClassifier = request.app.state.classifier
        return {
            "status": "ok",
            "version": __version__,
            "classes": clf.classes,
            "metrics": clf.metrics,
            "features": {"azure_tts": settings.azure_tts_enabled, "grammar": settings.enable_grammar},
        }

    @app.post("/api/predict/landmarks")
    def predict_landmarks(body: LandmarkRequest, request: Request):
        clf: SignClassifier = request.app.state.classifier
        hands = [np.asarray(h, dtype=np.float32) for h in body.hands]
        return clf.predict_landmarks(hands, body.width, body.height, body.top_k).to_dict()

    @app.post("/api/grammar")
    def grammar(body: TextRequest, request: Request):
        corrector: GrammarCorrector | None = request.app.state.grammar
        if corrector is None:
            return {"text": body.text, "corrected": False}
        try:
            return {"text": corrector.correct(body.text), "corrected": True}
        except Exception as exc:
            log.exception("grammar correction failed")
            raise HTTPException(503, f"Grammar correction unavailable: {exc}")

    @app.post("/api/speak")
    def speak(body: TextRequest):
        if not settings.azure_tts_enabled:
            raise HTTPException(503, "Azure speech is not configured; the app uses the browser's voice instead")
        try:
            audio = tts.synthesize(body.text, settings.azure_speech_key, settings.azure_speech_region,
                                   settings.azure_speech_voice)
        except tts.TTSError as exc:
            log.warning("TTS failed: %s", exc)
            raise HTTPException(502, str(exc))
        return Response(audio, media_type="audio/wav")

    # Model files for the browser: MediaPipe hand landmarker + the exported sign classifier.
    model_files = {
        "hand_landmarker.task": settings.hand_model_path,
        settings.sign_model_path.name: settings.sign_model_path,
        settings.sign_model_path.with_suffix(".bin").name: settings.sign_model_path.with_suffix(".bin"),
    }

    @app.get("/models/{name}", include_in_schema=False)
    def model_file(name: str):
        if name not in model_files:
            raise HTTPException(404, "Not found")
        return FileResponse(model_files[name], headers={"Cache-Control": "public, max-age=3600"})

    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = create_app()
