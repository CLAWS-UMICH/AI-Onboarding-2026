"""Drop train rows that are near-copies of a test row, so test scores stay honest.

usage: uv run python dedup.py <candidate.jsonl> [--out data/train.jsonl]

A train row is dropped if its MPNet cosine similarity to any test row with the
SAME label is >= THRESHOLD. Different-label matches are contrast pairs
("fan to rpm" vs "fan one rpm") and are kept on purpose.
"""
import argparse
import json

from sentence_transformers import SentenceTransformer, util

from common import HERE, TEST_SETS

THRESHOLD = 0.95  # 0.95+ = one word added/swapped; 0.90-0.95 = real paraphrases


def load(path):  # full rows, not common.load's (texts, labels): we rewrite rows as-is
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("candidate")
ap.add_argument("--out", default="data/train.jsonl")
args = ap.parse_args()

train = load(args.candidate)
tests = [row for name in TEST_SETS for row in load(HERE / "data" / f"test_{name}.jsonl")]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
sim = util.cos_sim(
    model.encode([r["text"] for r in train], convert_to_tensor=True),
    model.encode([r["text"] for r in tests], convert_to_tensor=True),
)

keep, dropped = [], []
for i, row in enumerate(train):
    same_label = [j for j, t in enumerate(tests) if t["intents"] == row["intents"]]
    best = max(sim[i, j].item() for j in same_label) if same_label else 0.0
    (dropped if best >= THRESHOLD else keep).append(row)

with open(args.out, "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in keep)
print(f"kept {len(keep)}, dropped {len(dropped)} (>= {THRESHOLD} to a same-label test row)")
for r in dropped:
    print("  -", r["text"], r["intents"])
