"""Check a training file: known labels, one label per row, no duplicates, enough rows."""
import json
import sys
from collections import Counter
from pathlib import Path

MIN_PER_INTENT = 30

labels = {x["intent"] for x in json.load(open(Path(__file__).with_name("intents.json")))}
seen, counts, problems = set(), Counter(), 0

for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    text, intents = row["text"].strip(), row["intents"]
    if len(intents) != 1 or intents[0] not in labels:
        print(f"line {n}: bad intents {intents}")
        problems += 1
    if text.lower() in seen:
        print(f"line {n}: duplicate text '{text}'")
        problems += 1
    seen.add(text.lower())
    counts.update(intents)

for label in sorted(labels):
    if counts[label] < MIN_PER_INTENT:
        print(f"{label}: only {counts[label]} rows (need {MIN_PER_INTENT})")
        problems += 1

print(f"{sum(counts.values())} rows, {problems} problems")
sys.exit(1 if problems else 0)
