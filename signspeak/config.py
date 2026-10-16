"""Runtime settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Real environment variables always win over values in .env.
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, *fallbacks: str, default: str | None = None) -> str | None:
    for key in (name, *fallbacks):
        value = os.getenv(key)
        if value:
            return value.strip()
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    hand_model_path: Path
    sign_model_path: Path
    frontend_dir: Path
    azure_speech_key: str | None
    azure_speech_region: str | None
    azure_speech_voice: str
    enable_grammar: bool
    cors_origins: tuple[str, ...]

    @property
    def azure_tts_enabled(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)


def get_settings() -> Settings:
    cors = _env("CORS_ORIGINS", default="") or ""
    return Settings(
        hand_model_path=_env_path("HAND_MODEL_PATH", PROJECT_ROOT / "models" / "hand_landmarker.task"),
        sign_model_path=_env_path("SIGN_MODEL_PATH", PROJECT_ROOT / "models" / "sign_mlp.json"),
        frontend_dir=_env_path("FRONTEND_DIR", PROJECT_ROOT / "frontend"),
        # API_KEY / REGION are older variable names, still accepted.
        azure_speech_key=_env("AZURE_SPEECH_KEY", "API_KEY"),
        azure_speech_region=_env("AZURE_SPEECH_REGION", "REGION"),
        azure_speech_voice=_env("AZURE_SPEECH_VOICE", default="en-IN-NeerjaNeural") or "en-IN-NeerjaNeural",
        enable_grammar=_env_bool("ENABLE_GRAMMAR", default=False),
        cors_origins=tuple(o.strip() for o in cors.split(",") if o.strip()),
    )
