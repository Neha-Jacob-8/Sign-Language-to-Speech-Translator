"""Optional grammar correction with the T5 "Gramformer" model described in the project report.

Fingerspelling produces bare words ("she not like coffee"); this rewrites them as a sentence
("She does not like coffee.").
Enable with ENABLE_GRAMMAR=1 after `pip install -r requirements-grammar.txt`.
The model (~900 MB) is downloaded from Hugging Face the first time it is used.
"""
from __future__ import annotations

import threading

MODEL_ID = "prithivida/grammar_error_correcter_v1"


class GrammarCorrector:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Grammar correction needs: pip install -r requirements-grammar.txt") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).eval()

    def correct(self, text: str) -> str:
        text = " ".join(text.split())
        if not text:
            return text
        import torch

        with self._lock:
            self._load()
            inputs = self._tokenizer("gec: " + text.lower(), return_tensors="pt")
            with torch.no_grad():
                out = self._model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
            fixed = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return fixed[:1].upper() + fixed[1:]
