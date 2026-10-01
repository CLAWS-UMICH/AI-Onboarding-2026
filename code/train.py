"""Step 1: fine-tune. Write your own version in your folder.

Outline (do it however you like):
- Load data/train.jsonl and hold out the same ~15% split as your baseline.
- Map labels to ids; train the encoder + head with cross-entropy.
- Report accuracy and macro-F1 on the held-out set; print every miss.
- Save to runs/: model weights (model.pt, not committed), label2id.json,
  and metrics.json with at least model, accuracy, macro_f1.
"""
