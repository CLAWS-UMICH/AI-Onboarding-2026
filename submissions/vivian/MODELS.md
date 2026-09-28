# Models you can use

Pick one encoder. `baseline.py` and `train.py` both use it, so you compare like with like. `predict.py` reads your choice back from `runs/metrics.json`.

| # | Model | Size | Hidden | Notes |
|---|---|---|---|---|
| 1 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | **Default.** Fastest; trains in minutes on a laptop CPU. |
| 2 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | 384 | Same size class, often more accurate on short text. |
| 3 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 110M | 768 | Biggest and usually most accurate, ~5x slower. Use a GPU/MPS if you have one. |

## Switch models
Set `INTENT_MODEL` before running baseline and train (skip it to use MiniLM):

```bash
export INTENT_MODEL=BAAI/bge-small-en-v1.5          # macOS / Linux
$env:INTENT_MODEL="BAAI/bge-small-en-v1.5"          # Windows PowerShell
set INTENT_MODEL=BAAI/bge-small-en-v1.5             # Windows Command Prompt

python ../../code/baseline.py
python ../../code/train.py
python ../../code/predict.py "open the nav menu"   # no INTENT_MODEL needed
```

Changed models? Re-run **both** baseline and train. The variable only lasts for that terminal; open a new one and set it again. Write the model you used in `notes.md`.

One difference: the baseline embeds sentences the way each model was published (bge uses its first token, the other two average all tokens), while fine-tuning always averages all tokens. Doesn't matter for beating the baseline, just don't be surprised if bge's baseline is relatively strong.

The 36 intent labels are in `intents.json` in this folder (a copy of `code/intents.json`; the validator reads the one in `code/`).
