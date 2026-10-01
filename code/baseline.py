"""Step 0: baseline. No fine-tuning. Write your own version in your folder.

Outline (do it however you like):
- Load data/train.jsonl, hold out ~15% (stratified, fixed seed).
- Embed sentences with a frozen encoder (sentence-transformers is installed).
- Fit a simple classifier (e.g. sklearn logistic regression) on the embeddings.
- Print a per-intent report and save it to baseline.txt (include the model name).
"""
