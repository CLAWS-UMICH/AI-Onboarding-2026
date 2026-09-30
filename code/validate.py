"""Check a training file: valid JSON, known labels, one label per row, no duplicates, enough rows."""
import json
import sys
from collections import Counter
from pathlib import Path

MIN_PER_INTENT = 30    # rows per intent; handwritten or synthetic both count

if len(sys.argv) != 2:
    sys.exit("usage: python validate.py data/train.jsonl")

intents_file = Path(__file__).with_name("intents.json")
labels = {x["intent"] for x in json.loads(intents_file.read_text(encoding="utf-8"))}
seen, total, problems = set(), Counter(), []

for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
        text, intents = row["text"].strip(), row["intents"]
    except (ValueError, KeyError, TypeError, AttributeError):
        problems.append(f'line {n}: not a valid row, expected {{"text": "...", "intents": ["label"]}}')
        continue
    if not isinstance(intents, list) or len(intents) != 1 or intents[0] not in labels:
        problems.append(f"line {n}: bad intents {intents!r} (need exactly one label from intents.json)")
        continue
    if text.lower() in seen:
        problems.append(f"line {n}: duplicate text '{text}'")
    seen.add(text.lower())
    total[intents[0]] += 1

for label in sorted(labels):
    if total[label] < MIN_PER_INTENT:
        problems.append(f"{label}: only {total[label]} rows (need {MIN_PER_INTENT})")

print("\n".join(problems))
print(f"{sum(total.values())} rows, {len(problems)} problems")
sys.exit(1 if problems else 0)
