import torch

try:
    # Import T5 classes; if SentencePiece is missing transformers will raise an error
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except Exception as e:
    msg = str(e)
    if "SentencePiece" in msg or "sentencepiece" in msg or "T5Tokenizer requires the SentencePiece" in msg:
        raise ImportError(
            "The transformers T5Tokenizer requires the 'sentencepiece' library which is not installed in your environment.\n"
            "Install it into your virtualenv with:\n\n"
            "    pip install sentencepiece\n\n"
            "If you're using a virtual environment, activate it first (for PowerShell):\n"
            "    .\\.venv\\Scripts\\Activate.ps1\n\n"
            "Then restart your Python process or terminal and try again."
        ) from e
    # Re-raise if it's a different error
    raise


class Gramformer:
    def __init__(self, models=1, use_gpu=False):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('prithivida/grammar_error_correcter_v1')
        self.tokenizer = T5Tokenizer.from_pretrained('prithivida/grammar_error_correcter_v1')
        self.model = self.model.to(self.device)

    def correct(self, input_sentence, max_candidates=1):
        input_text = "gec: " + input_sentence
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=7,
            num_return_sequences=max_candidates,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            early_stopping=True
        )

        corrected = set()
        for output in outputs:
            corrected.add(self.tokenizer.decode(output, skip_special_tokens=True))

        return corrected