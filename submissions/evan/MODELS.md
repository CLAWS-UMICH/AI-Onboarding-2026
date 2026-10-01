# Models you can use

Pick one encoder. Use the same one for your baseline and your fine-tune so you compare like with like.

| # | Model | Size | Hidden | Notes |
|---|---|---|---|---|
| 1 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | **Default.** Fastest; trains in minutes on a laptop CPU. |
| 2 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | 384 | Same size class, often more accurate on short text. |
| 3 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 110M | 768 | Biggest and usually most accurate, ~5x slower. Use a GPU/MPS if you have one. |

## Switch models
How you choose the model is up to you (a constant, a CLI flag, an env var). Changed models? Re-run **both** baseline and fine-tune. Write the model you used in `notes.md` and `runs/metrics.json`.

The 36 intent labels are in `intents.json` in this folder (a copy of `code/intents.json`; the validator reads the one in `code/`).
