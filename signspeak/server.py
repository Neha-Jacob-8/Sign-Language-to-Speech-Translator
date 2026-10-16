"""FastAPI backend: serves the web app and the recognition / speech API.

    uvicorn signspeak.server:app --reload
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from . import __version__, tts
from .config import Settings, get_settings
from .features import NUM_LANDMARKS
from .grammar import GrammarCorrector
from .landmarks import load_image
from .recognizer import SignRecognizer

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


def create_app(settings: Settings | None = None, recognizer: SignRecognizer | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rec = recognizer or SignRecognizer.load(settings.sign_model_path, settings.hand_model_path)
        rec.detector  # create MediaPipe on this thread now, not lazily inside a request worker thread
        app.state.recognizer = rec
        app.state.grammar = GrammarCorrector() if settings.enable_grammar else None
        log.info("Loaded sign model with %d classes", len(rec.classes))
        yield
        rec.close()  # MediaPipe's native threads otherwise keep the process from exiting

    app = FastAPI(title="SignSpeak: ISL to Speech", version=__version__, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/api/health")
    def health(request: Request):
        rec: SignRecognizer = request.app.state.recognizer
        return {
            "status": "ok",
            "version": __version__,
            "classes": rec.classes,
            "metrics": rec.metrics,
            "features": {"azure_tts": settings.azure_tts_enabled, "grammar": settings.enable_grammar},
        }

    @app.post("/api/predict/landmarks")
    def predict_landmarks(body: LandmarkRequest, request: Request):
        rec: SignRecognizer = request.app.state.recognizer
        hands = [np.asarray(h, dtype=np.float32) for h in body.hands]
        return rec.predict_landmarks(hands, body.width, body.height, body.top_k).to_dict()

    @app.post("/api/predict/image")
    async def predict_image(request: Request, file: UploadFile = File(...)):
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Image is larger than {settings.max_upload_mb} MB")
        try:
            image = load_image(io.BytesIO(data))
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "The file is not a readable image")
        rec: SignRecognizer = request.app.state.recognizer
        pred = await run_in_threadpool(rec.predict_image, image)
        return pred.to_dict()

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

    @app.get("/models/hand_landmarker.task", include_in_schema=False)
    def hand_model():
        # The browser loads the same MediaPipe model the server and training use.
        return FileResponse(settings.hand_model_path, media_type="application/octet-stream")

    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = create_app()
