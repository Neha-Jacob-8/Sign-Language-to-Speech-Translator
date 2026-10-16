from gramformer import Gramformer
import torch

# Instantiate the model. You can specify to correct (1) or highlight (0).
# You might need to run this twice on the first go to let it download the model.
gf = Gramformer(models = 1, use_gpu=False) # Use GPU if you have a compatible one

broken_sentences = [
    "What you doing?",
    "You not hungry",
    "Tomorrow will go school",
    "She not like coffee",
]

print("Correcting sentences...")
for sentence in broken_sentences:
    corrected_sentences = gf.correct(sentence, max_candidates=1)
    # gf.correct returns a set of possibilities
    corrected = list(corrected_sentences)[0] if corrected_sentences else sentence
    print(f"[Input]: {sentence}  ->  [Output]: {corrected}")
