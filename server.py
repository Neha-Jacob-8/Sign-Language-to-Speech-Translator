# server.py — Always display translation of live.txt; Save only archives a copy.
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import io, re, time, glob, json, shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# ───────────────────────── Paths ─────────────────────────
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# On Azure, APP_ROOT will be /home/site/wwwroot. Locally it falls back to this folder.
ROOT_DIR = Path(os.getenv("APP_ROOT", str(BASE_DIR)))

FRONTEND_DIR = ROOT_DIR / "frontend"
CKPT_DIR     = ROOT_DIR / "checkpoints"
CONFIG_JSON  = ROOT_DIR / "config.json"
TRANS_CKPT   = ROOT_DIR / "translator" / "isl2en_attention.pt"
SPEECH_DIR   = ROOT_DIR / "speech"

# Writable, persistent area (Azure: /home/site/data). Local default: ./persist
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", str(BASE_DIR / "persist")))
UPLOADS_DIR = PERSIST_DIR / "uploads"
SAVES_DIR   = UPLOADS_DIR / "text"      # archives (timestamped)
AUDIO_DIR   = UPLOADS_DIR / "audio"     # generated audio
LIVE_FILE   = SAVES_DIR / "live.txt"    # always overwritten
LIVE_META   = SAVES_DIR / "live_meta.json"

for p in (UPLOADS_DIR, SAVES_DIR, AUDIO_DIR):
    p.mkdir(parents=True, exist_ok=True)
# ───────────────────────── Device ─────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
def log(msg: str): print(f"[isl] {msg}", flush=True)
def _stamp() -> str: return time.strftime("%Y-%m-%d_%H%M%S")
def _sanitize(name: str) -> str:
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.\-]", "", name) or "img"

# ───────────────────────── Model (SmallSignCNN) ─────────────────────────
class SmallSignCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        def blk(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2)
            )
        self.f = nn.Sequential(
            blk(3,32), blk(32,64), blk(64,128), blk(128,256),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.h = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256,256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256,n_classes)
        )
    def forward(self,x):
        x = self.f(x)
        x = x.reshape(x.size(0), -1)
        return self.h(x)

def _val_score(p: Path) -> float:
    m = re.search(r"_val([0-9]+(?:\.[0-9]+)?)", p.stem)
    return float(m.group(1)) if m else -1.0

def _best_ckpt() -> Optional[Path]:
    cands = [Path(p) for p in glob.glob(str(CKPT_DIR / "*.pt"))]
    if not cands: return None
    cands.sort(key=lambda p: (_val_score(p), p.stat().st_mtime), reverse=True)
    return cands[0]

def load_classifier() -> tuple[nn.Module, list[str]]:
    ckpt = _best_ckpt()
    if not ckpt:
        raise RuntimeError(f"No checkpoints found in {CKPT_DIR}")
    obj = torch.load(str(ckpt), map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj or "classes" not in obj:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt}")
    classes = list(obj["classes"])
    model = SmallSignCNN(len(classes))
    model.load_state_dict(obj["model"], strict=False)
    model.to(DEVICE).eval()
    log(f"Loaded classifier {ckpt.name} ({len(classes)} classes) on {DEVICE}")
    return model, classes

# ───────────────────────── Eval TF ─────────────────────────
def _img_size() -> int:
    try:
        if CONFIG_JSON.exists():
            return int(json.loads(CONFIG_JSON.read_text()).get("img_size", 128))
    except Exception as e:
        log(f"config.json read failed: {e}")
    return 128

IMG_SIZE = _img_size()
EVAL_TF = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

@torch.inference_mode()
def _predict_char(model: nn.Module, classes: list[str], img: Image.Image) -> str:
    x = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    return classes[int(logits.argmax(1).item())].lower()

def _read_pil(u: UploadFile) -> Image.Image:
    raw = u.file.read()
    if not raw: raise ValueError("empty upload")
    return Image.open(io.BytesIO(raw)).convert("RGB")

def assemble_text(chars: List[str], gaps: List[bool], auto_space: bool) -> str:
    if not chars: return ""
    if auto_space: return " ".join(chars)
    out = [chars[0]]
    for i in range(len(chars)-1):
        if i < len(gaps) and gaps[i]:
            out.append(" ")
        out.append(chars[i+1])
    return "".join(out)

# ───────────────────────── Translator ─────────────────────────
_TRANSLATOR = None
def load_translator():
    global _TRANSLATOR
    if _TRANSLATOR is not None:
        return _TRANSLATOR
    if not TRANS_CKPT.exists():
        log(f"[translator] not found at {TRANS_CKPT}; identity used.")
        _TRANSLATOR = lambda s: s
        return _TRANSLATOR
    try:
        m = torch.jit.load(str(TRANS_CKPT), map_location="cpu")
        if hasattr(m, "eval"): m.eval()
        if hasattr(m, "to"):   m.to(DEVICE)
        log("[translator] TorchScript loaded")
        def _run(s: str) -> str:
            try:
                out = m(s)
                if isinstance(out, (list, tuple)) and out: out = out[0]
                return str(out)
            except Exception as e:
                log(f"[translator] script forward failed: {e}"); return s
        _TRANSLATOR = _run; return _TRANSLATOR
    except Exception as e:
        log(f"[translator] load failed: {e}")
        _TRANSLATOR = lambda s: s; return _TRANSLATOR

# ───────────────────────── App ─────────────────────────
app = FastAPI(title="ISL: live.txt display flow")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

_MODEL, _CLASSES = load_classifier()
log(f"img_size={IMG_SIZE}")

# ───────────────────────── API ─────────────────────────

@app.post("/api/recognize_live")
async def recognize_live(
    files: List[UploadFile] = File(...),
    gaps: str = Form("[]"),
    auto_space: str = Form("false")
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    try:
        gaps_list = json.loads(gaps) if gaps else []
        gaps_list = [bool(x) for x in gaps_list]
    except Exception:
        gaps_list = []
    auto = str(auto_space).lower() == "true"

    per_image: List[str] = []
    for old in UPLOADS_DIR.glob("temp_*.jpg"):
        try: old.unlink()
        except Exception: pass

    for i, f in enumerate(files):
        try:
            img = _read_pil(f)
            temp_path = UPLOADS_DIR / f"temp_{i}.jpg"
            img.save(temp_path)
            pred_char = _predict_char(_MODEL, _CLASSES, img)
            per_image.append(pred_char)
        except Exception as e:
            log(f"[predict] error on image {i}: {e}")
            per_image.append("?")

    text = assemble_text(per_image, gaps_list, auto).strip()
    if not text:
        raise HTTPException(500, "Failed to assemble text from images")

    LIVE_FILE.write_text(text, encoding="utf-8")
    LIVE_META.write_text(json.dumps({
        "per_image": per_image, "gaps": gaps_list, "auto_space": auto, "last_updated": _stamp()
    }, indent=2), encoding="utf-8")

    translator = load_translator()
    try:
        live_text = LIVE_FILE.read_text(encoding="utf-8")
        translated = translator(live_text)
    except Exception as e:
        log(f"[translator] call failed: {e}")
        translated = live_text

    return JSONResponse({
        "ok": True, "recognized_chars": per_image,
        "live_path": LIVE_FILE.as_posix(), "translation": translated
    })

@app.get("/api/translate_live")
def translate_live():
    if not LIVE_FILE.exists():
        return JSONResponse({"ok": False, "detail": "live.txt not found"}, status_code=404)
    translator = load_translator()
    text = LIVE_FILE.read_text(encoding="utf-8")
    try:
        out = translator(text)
    except Exception:
        out = text
    return JSONResponse({"ok": True, "translation": out, "source_path": LIVE_FILE.as_posix()})

@app.post("/api/save_live")
def save_live():
    if not LIVE_FILE.exists():
        raise HTTPException(400, "No live text to save yet. Click Recognize first.")
    ts = _stamp()
    out_path = SAVES_DIR / f"{ts}.txt"
    shutil.copy2(LIVE_FILE, out_path)
    return JSONResponse({"ok": True, "saved_path": out_path.as_posix()})

@app.get("/api/saved_texts")
def list_saved_texts(limit: int = 20):
    files = sorted(
        [p for p in SAVES_DIR.glob("*.txt") if p.name != LIVE_FILE.name],
        key=lambda p: p.stat().st_mtime, reverse=True
    )[:limit]
    return JSONResponse({"count": len(files), "files": [f.as_posix() for f in files]})

# ──────────────── SPEECH ENDPOINT (NEW) ────────────────
@app.post("/api/speech")
async def generate_speech():
    """Convert current translated text (live.txt) to speech using Azure Cognitive Services."""
    try:
        if not LIVE_FILE.exists():
            raise HTTPException(404, "live.txt not found")

        # Dynamically import the Azure speech synthesis module
        from speech.speech import synthesize_speech  # <-- implement in speech/speech.py

        text = LIVE_FILE.read_text(encoding="utf-8")
        if not text.strip():
            raise HTTPException(400, "live.txt is empty")

        # Generate output file path
        out_path = AUDIO_DIR / f"{_stamp()}_speech.wav"
        synthesize_speech(text, out_path)

        log(f"[speech] Generated: {out_path.name}")
        return FileResponse(out_path, media_type="audio/wav", filename=out_path.name)
    except Exception as e:
        log(f"[speech] error: {e}")
        raise HTTPException(500, f"Speech generation failed: {e}")

# Serve frontend
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
