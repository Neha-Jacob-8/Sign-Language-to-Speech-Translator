"""Text-to-speech through Azure AI Speech.

Optional: when no key is configured, the web app speaks with the browser's built-in voice instead.
"""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

import requests

OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"


class TTSError(RuntimeError):
    pass


def build_ssml(text: str, voice: str) -> str:
    lang = "-".join(voice.split("-")[:2]) or "en-IN"  # "en-IN-NeerjaNeural" -> "en-IN"
    # The text comes from users, so it must be escaped before it goes into XML.
    return (
        f"<speak version='1.0' xml:lang={quoteattr(lang)}>"
        f"<voice name={quoteattr(voice)}>{escape(text)}</voice>"
        "</speak>"
    )


def synthesize(text: str, key: str, region: str, voice: str, timeout: float = 15.0) -> bytes:
    """Return WAV audio for `text`, or raise TTSError."""
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
        "User-Agent": "signspeak",
    }
    try:
        resp = requests.post(url, headers=headers, data=build_ssml(text, voice).encode("utf-8"), timeout=timeout)
    except requests.RequestException as exc:
        raise TTSError(f"could not reach Azure Speech: {exc}") from exc
    if resp.status_code != 200:
        raise TTSError(f"Azure Speech returned HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.content
