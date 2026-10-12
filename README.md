# Sign Language to Speech Translator

This repository serves a FastAPI app that:
- classifies uploaded sign images into character sequences,
- assembles and translates the current sequence from `live.txt`,
- optionally converts the text to speech using Azure Cognitive Services,
- hosts a browser UI from `frontend/index.html`.

## Current runtime architecture

- Backend entrypoint: `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/server.py`
- Frontend: `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/frontend/index.html`
- Speech module: `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/speech/speech.py`
- Classifier checkpoints: `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/checkpoints/*.pt`
- Translator checkpoint: `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/translator/isl2en_attention.pt`

## Local setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure Azure speech credentials (required for `/api/speech`):
   - `API_KEY`
   - `REGION`
   
   You can put them in:
   - `secure.env` (preferred by `speech/speech.py`), or
   - `.env`, or
   - environment variables (for cloud/App Service).
3. Run the app:
   ```bash
   uvicorn server:app --reload
   ```
4. Open `http://127.0.0.1:8000`.

## Data and persistence behavior

- The app writes generated files under `PERSIST_DIR`:
  - Default: `persist/` inside the repo.
  - Override with environment variable `PERSIST_DIR`.
- Runtime files:
  - `persist/uploads/text/live.txt` (always overwritten by **Recognize**)
  - `persist/uploads/text/live_meta.json` (recognition metadata)
  - `persist/uploads/text/<timestamp>.txt` (saved archives from **Save**)
  - `persist/uploads/audio/<timestamp>_speech.wav` (speech output)

## API behavior implemented today

- `POST /api/recognize_live`  
  Accepts uploaded images (`files`), optional `gaps` and `auto_space`; updates `live.txt` and returns recognized characters plus a translated string.
- `GET /api/translate_live`  
  Reads current `live.txt` and returns translation.
- `POST /api/save_live`  
  Copies current `live.txt` into a timestamped `.txt` file.
- `GET /api/saved_texts`  
  Lists most recent saved text files.
- `POST /api/speech`  
  Generates WAV from current `live.txt` content using Azure Speech.

## Training and developer docs

See `/home/runner/work/Sign-Language-to-Speech-Translator/Sign-Language-to-Speech-Translator/docs/developer.md` for:
- `imagetraining.py` workflow details,
- expected `config.json` usage,
- deployment workflow notes from `.github/workflows/main_signtospeech.yml`.
