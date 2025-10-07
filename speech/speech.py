import requests
from pathlib import Path
import os
from dotenv import load_dotenv

# ───────────────────────── Paths ─────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[1]
LIVE_FILE = BASE_DIR / "uploads" / "text" / "live.txt"
AUDIO_DIR = BASE_DIR / "uploads" / "audio"
SECURE_ENV = BASE_DIR / "secure.env"  # ✅ your env file

# ───────────────────────── Load environment ─────────────────────────
# Prefer secure.env → fallback to .env
if SECURE_ENV.exists():
    load_dotenv(SECURE_ENV)
else:
    load_dotenv(BASE_DIR / ".env")

# For cloud (Azure App Service), these are automatically picked up from App Settings
API_KEY = os.getenv("API_KEY")
REGION  = os.getenv("REGION")

if not API_KEY or not REGION:
    raise RuntimeError("❌ Missing Azure Speech credentials. Check secure.env, .env, or Azure App Settings.")

# ───────────────────────── Azure URLs ─────────────────────────
TOKEN_URL = f"https://{REGION}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
TTS_URL   = f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

# ───────────────────────── Speech Synth ─────────────────────────
def synthesize_speech(text: str, output_path: Path | str = "output.wav") -> str:
    """Convert text to speech using Azure Cognitive Services and save to output_path."""
    try:
        # 1️⃣ Get access token
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        response = requests.post(TOKEN_URL, headers=headers)
        response.raise_for_status()
        access_token = response.text

        # 2️⃣ Prepare SSML body
        ssml = f"""
        <speak version='1.0' xml:lang='en-IN'>
            <voice xml:lang='en-IN' xml:gender='Female' name='en-IN-NeerjaNeural'>
                {text}
            </voice>
        </speak>
        """

        # 3️⃣ Make TTS request
        tts_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            "User-Agent": "python-tts"
        }

        tts_response = requests.post(TTS_URL, headers=tts_headers, data=ssml)
        tts_response.raise_for_status()

        # 4️⃣ Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tts_response.content)

        print(f"[speech] Saved audio → {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"[speech] Error: {e}")
        raise RuntimeError(f"Speech synthesis failed: {e}")

# ───────────────────────── Local test ─────────────────────────
if __name__ == "__main__":
    if not LIVE_FILE.exists():
        raise RuntimeError(f"live.txt not found at: {LIVE_FILE}")

    text = LIVE_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("live.txt is empty — run recognition first.")

    from time import strftime
    out_path = AUDIO_DIR / f"{strftime('%Y-%m-%d_%H%M%S')}_speech.wav"
    synthesize_speech(text, out_path)