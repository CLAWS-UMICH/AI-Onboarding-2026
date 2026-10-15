"""Step 0: frozen MPNet embeddings + logistic regression. The number to beat.

usage: uv run python baseline.py [--final]
Writes baseline.txt (full report) and runs/baseline/metrics.json.
--final also scores test_final: only run it once, at the very end.
"""
import argparse
import contextlib
import io
import json

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from common import HERE, load, load_test, score, split

MODEL = "sentence-transformers/all-mpnet-base-v2"

ap = argparse.ArgumentParser()
ap.add_argument("--final", action="store_true")
args = ap.parse_args()

texts, labels = load(HERE / "data" / "train.jsonl")
Xtr, Xval, ytr, yval = split(texts, labels)

# Frozen: the encoder never changes, it just turns each sentence into 768 numbers.
encoder = SentenceTransformer(MODEL)
embed = lambda xs: encoder.encode(xs, normalize_embeddings=True, batch_size=64)
clf = LogisticRegression(max_iter=2000).fit(embed(Xtr), ytr)

evals = {"val": (Xval, yval)}
evals |= {name: load_test(name) for name in ["clean", "attack"] + (["final"] if args.final else [])}

report = io.StringIO()
metrics = {"model": MODEL, "method": "frozen + logistic regression"}
with contextlib.redirect_stdout(report):
    print(f"model: {MODEL} (frozen) + LogisticRegression")
    print(f"train {len(Xtr)} / val {len(Xval)}")
    for name, (x, y) in evals.items():
        metrics[name] = score(x, y, clf.predict(embed(x)), title=name)

print(report.getvalue())
(HERE / "baseline.txt").write_text(report.getvalue(), encoding="utf-8")
out = HERE / "runs" / "baseline"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
print({k: v for k, v in metrics.items() if isinstance(v, dict)})
