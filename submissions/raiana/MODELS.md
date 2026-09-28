# Models you can use

Pick one encoder. `baseline.py` and `train.py` both use it, so the baseline stays a fair comparison. `predict.py` reads your choice back from `runs/metrics.json`.

| # | Model | Size | Hidden | Notes |
|---|---|---|---|---|
| 1 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | **Default.** Fastest; trains in minutes on a laptop CPU. |
| 2 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | 384 | Same size class, often more accurate on short text. |
| 3 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 110M | 768 | Biggest and usually most accurate, ~5x slower. Use a GPU/MPS if you have one. |

## Switch models
Set `MODEL` before running baseline and train (skip it to use MiniLM):

```bash
export MODEL=BAAI/bge-small-en-v1.5          # macOS / Linux
$env:MODEL="BAAI/bge-small-en-v1.5"          # Windows PowerShell

python ../../code/baseline.py
python ../../code/train.py
python ../../code/predict.py "open the nav menu"   # no MODEL needed
```

Changed models? Re-run **both** baseline and train. Write the model you used in `notes.md`.

The 36 intent labels are in `intents.json` in this folder (a copy of `code/intents.json`; the validator reads the one in `code/`).
