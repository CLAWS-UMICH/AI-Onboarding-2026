"""Speech-to-text noise: copy some clean train rows and swap domain words for realistic Whisper mishearings.

usage: uv run python augment.py --out <file.jsonl> [--per-intent 6]

Reproducible (fixed seed). Output rows are tagged "source": "augment" and still need dedup.py + validate.py.
"""
import argparse
import json
import random
import re

from common import HERE

# Domain word -> how speech-to-text tends to write it. Meaning is unchanged, so the label is too.
# Informed by the error kinds seen on test_attack (spelled-out letters, homophones, split words);
# test_final_attack is written blind and is the honest check that this generalizes.
# Careful: "two"->"too/to", "one"->"won" and "scrubber a"->"scrubber eh" can erase the word that
# separates primary/secondary or A/B; the round-1 critic dropped those rows (see notes.md).
SUBS = {
    r"\bo2\b": ["oh two", "oh too", "o two", "o 2"],
    r"\bco2\b": ["see oh two", "c o two", "c o 2", "sea oh two"],
    r"\btwo\b": ["too", "to"],
    r"\bone\b": ["won"],
    r"\bprimary\b": ["prime", "pri", "primery"],
    r"\bsecondary\b": ["sec", "second", "secondery"],
    r"\bpressure\b": ["press", "presher"],
    r"\boxygen\b": ["oxy", "ox"],
    r"\bvitals\b": ["vidals", "vital"],
    r"\buia\b": ["you eye ay", "u i a"],
    r"\bhelmet\b": ["helmut"],
    r"\bscrubber a\b": ["scrubber ay", "scrubber eh"],
    r"\bscrubber b\b": ["scrubber bee", "scrubber be"],
    r"\bscrubber\b": ["scrubba"],
    r"\bbattery\b": ["batt", "battry"],
    r"\bcoolant\b": ["cool ant"],
    r"\bwarnings\b": ["warnins", "warning"],
    r"\bcoordinates\b": ["cord nets", "coordinate"],
    r"\bwaypoint\b": ["way point"],
    r"\btemperature\b": ["temprature", "temp"],
    r"\brpm\b": ["r p m", "rpms"],
}


def mishear(text, rng):
    """Apply 1-2 substitutions that match; None if nothing matched."""
    hits = [p for p in SUBS if re.search(p, text, re.IGNORECASE)]
    if not hits:
        return None
    for p in rng.sample(hits, k=min(len(hits), rng.choice([1, 2]))):
        text = re.sub(p, rng.choice(SUBS[p]), text, count=1, flags=re.IGNORECASE)
    return text


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-intent", type=int, default=6)
args = ap.parse_args()

rng = random.Random(0)
rows = [json.loads(line) for line in open(HERE / "data" / "train.jsonl", encoding="utf-8") if line.strip()]
seen = {r["text"].lower() for r in rows}
by_intent = {}
for r in rows:
    by_intent.setdefault(r["intents"][0], []).append(r)

out = []
for intent, group in by_intent.items():
    rng.shuffle(group)
    made = 0
    for r in group:
        new = mishear(r["text"], rng)
        if new and new.lower() not in seen:
            seen.add(new.lower())
            out.append({"text": new, "intents": [intent], "source": "augment"})
            made += 1
        if made == args.per_intent:
            break

with open(args.out, "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in out)
print(f"wrote {len(out)} rows to {args.out}")
