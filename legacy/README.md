# Legacy: version 1 (pixel CNN)

This folder keeps the first version of the project, as described in the October 2025 report, for reference. The current app does not use it.

| Folder | Contents |
|---|---|
| `cnn/` | `SmallSignCNN` training pipeline: dataset generation (`dataset.py`), 5-fold × 2 cross-validation training (`imagetraining.py`), test script, helper scripts, the 10 fold checkpoints and their reports |
| `v1_app/` | Original FastAPI server, single-page frontend and Azure speech module |
| `DL_Translator/` | Gramformer (T5) grammar-correction experiments, now integrated as `signspeak/grammar.py` |

## Why it was replaced

1. **One image per class.** `one_from_indian_dataset.py` copied a single image per class from the ISL dataset into `one_from_each/`. `dataset.py` then generated ~1,190 augmented copies of each one, giving the 42,744-image training set.
2. **Leaky validation.** Cross-validation and the `test/` split were drawn from those augmented copies of the same seed images, so the 99.9–100% fold accuracies measured memorisation of the seeds, not generalisation.
3. **Real photos failed.** The CNN learned the dataset's dark background and lighting. Two phone photos of different signs were both predicted as "5" (`uploads/text/live_meta.json`).
4. **Class "0"** came from an ASL silhouette dataset (50×50 binary masks), unlike every other class.
5. **The "translator"** (`translator/isl2en_attention.pt` on GitHub) was a raw LSTM state dict with no vocabulary. Loading it with `torch.jit.load` always failed, so the server silently returned its input unchanged.

Fixes made to the files kept here:
- `cnn/test.py` used to print `(acc - 0.01) * 100`, reporting one percentage point less than measured. It now prints the measured accuracy.
- The Azure key that was hard-coded in `speech.py` was removed; the key is now read from the environment.

The scripts still contain the original absolute paths and are kept as a record of the method, not as runnable code.
