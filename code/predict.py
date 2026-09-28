"""Load runs/model.pt and classify text.
  python predict.py "open the nav menu"     one sentence
  python predict.py --peer evan             score your model on evan's pushed data
"""
import json
import re
import subprocess
import sys

import torch
from transformers import AutoTokenizer

from model import MODEL_NAME, IntentModel

MIN_CONF = 0.5   # below this, answer "unhandled" instead of guessing

label2id = json.load(open("runs/label2id.json", encoding="utf-8"))
labels = sorted(label2id, key=label2id.get)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = IntentModel(len(labels))
model.load_state_dict(torch.load("runs/model.pt", map_location="cpu"))
model.eval()

SLOT_INTENTS = {"Get_coordinates", "Set_navigation_target",
                "Add_waypoint", "Delete_waypoint", "Complete_task"}
# Naive: the slot is whatever follows the last trigger word. Improve this.
TRIGGER = re.compile(r"\b(?:to|named|called|for|of|waypoint|task)\s+", re.I)


def extract_slot(text):
    parts = TRIGGER.split(text.strip().rstrip(".?!"))
    return parts[-1].strip() if len(parts) > 1 else None


def predict(text):
    with torch.no_grad():
        probs = model(tokenizer([text], return_tensors="pt")).softmax(1)[0]
    conf, idx = probs.max(0)
    intent = labels[idx.item()] if conf.item() >= MIN_CONF else "unhandled"
    result = {"text": text, "selected_intent": intent, "confidence": round(conf.item(), 3)}
    if intent in SLOT_INTENTS:
        result["slot"] = extract_slot(text)
    return result


def evaluate_peer(name):
    # Read the peer's file straight from git, so nothing lands in your folder.
    ref = f"origin/onboarding/{name}:submissions/{name}/data/train.jsonl"
    data = subprocess.run(["git", "show", ref], capture_output=True, check=True).stdout
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    wrong = 0
    for row in rows:
        got = predict(row["text"])["selected_intent"]
        if got != row["intents"][0]:
            wrong += 1
            print(f"MISS  '{row['text']}'  expected {row['intents'][0]}  got {got}")
    print(f"accuracy on {name}'s data: {1 - wrong / len(rows):.3f} ({len(rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--peer":
        evaluate_peer(sys.argv[2])
    elif len(sys.argv) > 1:
        print(json.dumps(predict(" ".join(sys.argv[1:]))))
    else:
        sys.exit(__doc__)
