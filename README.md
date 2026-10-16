# SignSpeak: Indian Sign Language to Speech

SignSpeak turns **Indian Sign Language (ISL) fingerspelling** into text and speech. Sign letters and digits in front of a webcam (or upload photos), watch them become words, and have the message spoken aloud.

**Live demo: https://neha-jacob-8.github.io/Sign-Language-to-Speech-Translator/**

![Live camera mode](docs/screenshot-live.png)

- **Live camera mode:** hands are tracked in real time. Hold a sign steady to type it, and lower your hands to end a word.
- **Photo mode:** upload a sequence of photos, one sign each, and get the spelled text back with the detected hand skeleton drawn on every photo.
- **Speech:** the browser's built-in voice, or an Azure neural voice (`en-IN`) when the Python server is configured with a key.
- **Private:** recognition runs entirely in the browser, so video and photos never leave the device.
- **Optional grammar correction** with a T5 transformer (Gramformer), via the Python server: `she not like coffee` → `She does not like coffee.`
- Recognises **35 static ISL signs**: digits 1–9 and letters A–Z.

## How it works

```mermaid
flowchart LR
    A[Camera frame<br/>or photo] --> B[MediaPipe Hand Landmarker<br/>21 joints per hand]
    B --> C[Feature vector<br/>wrist-relative, scale-normalised,<br/>two hands ordered left→right]
    C --> D[MLP classifier<br/>trained in PyTorch, 35 classes]
    D --> E[Stabiliser<br/>moving average + hold-to-type]
    E --> F[Message] --> G[Speech]
```

1. **Hand landmarks.** Google's MediaPipe Hand Landmarker (a pretrained CNN) finds up to two hands and returns 21 3-D joint positions per hand.
2. **Features** ([signspeak/features.py](signspeak/features.py), [frontend/classifier.js](frontend/classifier.js)). Each hand is expressed relative to its wrist and divided by its palm length, so position in the frame, distance from the camera and image resolution don't matter. For two-handed signs the hands are ordered left-to-right and the vector between the wrists is added (131 values in total).
3. **Classifier** ([signspeak/model.py](signspeak/model.py)). A 3-layer MLP (256-256-128, BatchNorm, GELU, dropout 0.3), trained in PyTorch with label smoothing and on-the-fly augmentation: mirroring (left-handed signers, selfie cameras), ±90° rotation (tilted hands, sideways phone photos), scaling and joint jitter. After training, the network is exported with BatchNorm folded into the linear layers (`models/sign_mlp.json` + `.bin`, 0.5 MB) and runs in the browser with no ML framework. The tests check that the browser and Python implementations produce the same output.
4. **Stabiliser** ([frontend/app.js](frontend/app.js)). Predictions are smoothed with an exponential moving average. A letter is typed once it stays above the confidence threshold for the hold time (1 s by default), and a space is added when the hands are lowered.

### Why hand landmarks instead of raw pixels?

A CNN trained directly on the dataset images learns the dataset's background, lighting and signer as much as the hand shape. All 42k images show one person in front of a dark background, so such a model scores close to 100% on the dataset but fails on photos taken anywhere else. Hand landmarks remove background, lighting, skin tone and clothing before classification, so a model trained on one person transfers far better to others.

## Results

Trained on the ISL dataset (42,373 images in which a hand was detected; 35 classes). The images are consecutive video frames, so neighbouring files are near-duplicates. To avoid leaking those duplicates between train and test sets, the split assigns **contiguous blocks of 50 frames** to train, validation or test (70/15/15) rather than individual images.

| Evaluation | Result |
|---|---|
| Held-out frames from the dataset (6,996 images, blocked split) | **99.97%** accuracy, macro-F1 0.9997 |
| Photos of **five other people**, real backgrounds, mostly sideways (digits 1–9, 45 photos) | **57.8%** overall; **26 / 30 = 87%** on the digits these signers form the same way as the dataset |

The 13% gap on the cross-signer set comes from digits **6, 7 and 9**: those signers use different handshapes for them than the dataset signer (for example, a thumbs-up fist for 6). This is a data limitation, not a model error. See [reports/landmark_model/](reports/landmark_model/) for the confusion matrix, per-class report and training curves.

<p align="center"><img src="reports/landmark_model/training_curves.png" width="720" alt="Training curves"></p>

### Limitations

- **One signer in the training data.** Regional and personal variants of a sign are not recognised. More signers would help more than a bigger model would.
- **Static signs only.** Letters that involve motion are recognised from their hand shape alone.
- **No "0".** The dataset has no ISL zero.
- **Fingerspelling, not full ISL.** Word-level signs and ISL grammar are out of scope.

## Run it locally

Requires Python 3.10+ and a webcam. The trained model is included.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py                     # → http://127.0.0.1:8000
```

Click **Start camera**. Browsers only allow camera access on `localhost` or `https://`.

Keyboard shortcuts (when not typing in the message box): <kbd>Space</kbd> adds a space, <kbd>Backspace</kbd> deletes, <kbd>Enter</kbd> speaks.

### Optional server features

The Python server adds what a static site can't. Copy `.env.example` to `.env` and set:

| Variable | Purpose |
|---|---|
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Azure neural voice. Without these, the browser voice is used. |
| `AZURE_SPEECH_VOICE` | Voice name, default `en-IN-NeerjaNeural` |
| `ENABLE_GRAMMAR=1` | T5 grammar correction. First run `pip install -r requirements-grammar.txt`; the model (~900 MB) downloads on first use. |
| `CORS_ORIGINS` | Comma-separated origins, if the frontend is hosted separately |

## Training

```bash
pip install -r requirements-dev.txt

# 1. Dataset: one folder per sign, e.g. Indian/A/*.jpg, Indian/7/*.jpg
python -m training.extract_landmarks --data Indian          # → data/landmarks.npz (~90 s for 44k images)

# 2. Train + evaluate (writes models/sign_mlp.pt, models/sign_mlp.json/.bin and reports/landmark_model/)
python -m training.train --external Numbers                  # ~30 s on a laptop CPU

# 3. Test on any folder of labelled photos (ideally a person not in the training set)
python -m training.evaluate my_photos --show-errors
```

The dataset is the Kaggle *Indian Sign Language (ISL)* image dataset (128×128 images, ~1,200 per class). It is not included in this repository because of its size.

**To improve accuracy for new users**, add a few hundred photos of more signers to the class folders and re-run steps 1–2. Because the model works on landmarks, even phone photos taken against any background are useful training data.

## Deployment

Every push to `main` runs the tests and publishes the app to GitHub Pages ([.github/workflows/ci.yml](.github/workflows/ci.yml)). The site is static: `frontend/` plus the three model files. Any static host works the same way.

## API

When the Python server is running:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | – | status, classes, model metrics, enabled features |
| POST | `/api/predict/landmarks` | `{"hands": [[[x,y,z] ×21] ×≤2], "width", "height", "top_k"}`: normalised MediaPipe landmarks | `{label, confidence, top: [{label, prob}], num_hands}` |
| POST | `/api/grammar` | `{"text"}` | `{text, corrected}` |
| POST | `/api/speak` | `{"text"}` | `audio/wav` (503 if Azure is not configured) |

Interactive docs are at `/docs` while the server is running.

## Project structure

```
frontend/                web app: index.html, app.js, classifier.js, styles.css, sign reference images
models/                  hand_landmarker.task (MediaPipe), sign_mlp.json/.bin (classifier), sign_mlp.pt (PyTorch checkpoint)
signspeak/               Python package
  features.py            landmarks → feature vector
  classifier.py          NumPy inference on the exported model
  model.py               PyTorch MLP, checkpoint and export
  landmarks.py           MediaPipe hand detection for images (training)
  server.py              FastAPI server (optional)
  tts.py, grammar.py     Azure speech, T5 grammar correction
  config.py              settings from environment / .env
training/                extract_landmarks.py, train.py, evaluate.py
reports/landmark_model/  metrics, confusion matrix, training curves
tests/                   pytest suite, incl. browser-vs-Python classifier check
server.py                entry point: python server.py
```

## Tests

```bash
pytest
```

32 tests cover feature invariances, hand ordering, augmentation, the exported model against PyTorch, the browser classifier against Python (needs Node.js), the API and SSML escaping.

## Team

Group A16, 22AIE304 Deep Learning, Amrita School of Artificial Intelligence, Coimbatore:
Aashika Arunkumar, Diya Deepak, Diya Dinesh, Neha Jacob.
