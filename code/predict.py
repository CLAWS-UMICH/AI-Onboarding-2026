"""Load runs/model.pt and classify text.
  python predict.py "open the nav menu"     one sentence
"""
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer

if not os.path.exists("runs/metrics.json"):
    sys.exit("No runs/metrics.json. Run python ../../code/train.py from your folder first.")
# Use the encoder train.py actually used, not whatever INTENT_MODEL is set to now.
metrics = json.load(open("runs/metrics.json", encoding="utf-8"))
os.environ["INTENT_MODEL"] = metrics.get("model", "sentence-transformers/all-MiniLM-L6-v2")
sys.path.append(os.path.join("..", "..", "code"))  # lets a copy in your folder find model.py
from model import MODEL_NAME, IntentModel  # noqa: E402

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(predict(" ".join(sys.argv[1:]))))
    else:
        sys.exit(__doc__)
