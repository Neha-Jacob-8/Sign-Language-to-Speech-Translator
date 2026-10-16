import os
import requests
from pathlib import Path

# Azure Speech API credentials
API_KEY = os.getenv("AZURE_SPEECH_KEY", "")
REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")
TOKEN_URL = f"https://{REGION}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
TTS_URL = f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

def synthesize_speech(text: str, output_path: Path | str = "output.wav") -> str:
    """
    Convert text to speech using Azure Cognitive Services and save to output_path.
    Returns path to saved file.
    """
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

# ✅ Optional: local test mode
if __name__ == "__main__":
    synthesize_speech("Hello, this is a test of Azure Cognitive Services.", "output.wav")
