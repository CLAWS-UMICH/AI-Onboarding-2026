"""Load runs/model.pt and classify text. Usage: python predict.py "open the nav menu" """
import json
import re
import sys

import torch
from transformers import AutoTokenizer

from model import MODEL_NAME, IntentModel

label2id = json.load(open("runs/label2id.json"))
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
    result = {"text": text, "selected_intent": labels[idx.item()],
              "confidence": round(conf.item(), 3)}
    if result["selected_intent"] in SLOT_INTENTS:
        result["slot"] = extract_slot(text)
    return result


if __name__ == "__main__":
    print(json.dumps(predict(" ".join(sys.argv[1:]))))
