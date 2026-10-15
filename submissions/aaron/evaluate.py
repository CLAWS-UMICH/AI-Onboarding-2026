"""Score the saved model (runs/mpnet_ft/model.pt) on test sets, no retraining.

usage: uv run python evaluate.py attack final_attack     # names -> data/test_<name>.jsonl
Prints raw scores (top intent) and with predict.py's unhandled threshold.
"""
import sys

from common import HERE, LABELS, load, score
from predict import THRESHOLD, Predictor

predictor = Predictor()
results = {}
for name in sys.argv[1:]:
    texts, labels = load(HERE / "data" / f"test_{name}.jsonl")
    probs = predictor.probs(texts)
    conf, idx = probs.max(dim=1)
    raw = [LABELS[i] for i in idx.tolist()]
    gated = [p if c >= THRESHOLD else "unhandled" for p, c in zip(raw, conf.tolist())]
    results[name] = {"raw": score(texts, labels, raw, title=f"{name} raw"),
                     f"threshold {THRESHOLD}": score(texts, labels, gated, title=f"{name} threshold {THRESHOLD}")}
print()
for name, r in results.items():
    print(name, r)
