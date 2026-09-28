"""Step 1: fine-tune the encoder (MODEL env var) + linear head on data/train.jsonl, save to runs/."""
import json
import os
import random

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from model import MODEL_NAME, IntentModel, best_device

EPOCHS, BATCH, LR = 10, 16, 5e-5
random.seed(0)
torch.manual_seed(0)
device = best_device()

rows = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8") if line.strip()]
labels = sorted({r["intents"][0] for r in rows})
label2id = {label: i for i, label in enumerate(labels)}
texts = [r["text"] for r in rows]
ys = [label2id[r["intents"][0]] for r in rows]
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, ys, test_size=0.15, stratify=ys, random_state=0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = IntentModel(len(labels)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def encode(batch_texts):
    enc = tokenizer(batch_texts, padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def predict_ids(batch_texts):
    model.eval()
    with torch.no_grad():
        return [i for s in range(0, len(batch_texts), BATCH)
                for i in model(encode(batch_texts[s:s + BATCH])).argmax(1).tolist()]


for epoch in range(1, EPOCHS + 1):
    model.train()
    order = list(range(len(X_tr)))
    random.shuffle(order)
    losses = []
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        logits = model(encode([X_tr[i] for i in idx]))
        target = torch.tensor([y_tr[i] for i in idx], device=device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    pred = predict_ids(X_te)
    print(f"epoch {epoch}  loss {sum(losses) / len(losses):.3f}  "
          f"acc {accuracy_score(y_te, pred):.3f}  "
          f"macro-F1 {f1_score(y_te, pred, average='macro'):.3f}")

# Show every mistake on the held-out set: this is where your data needs work.
for text, gold, guess in zip(X_te, y_te, pred):
    if gold != guess:
        print(f"MISS  '{text}'  expected {labels[gold]}  got {labels[guess]}")

os.makedirs("runs", exist_ok=True)
torch.save(model.state_dict(), "runs/model.pt")
json.dump(label2id, open("runs/label2id.json", "w", encoding="utf-8"), indent=2)
json.dump({"model": MODEL_NAME, "accuracy": accuracy_score(y_te, pred),
           "macro_f1": f1_score(y_te, pred, average="macro"),
           "train_rows": len(X_tr), "test_rows": len(X_te),
           "epochs": EPOCHS, "batch": BATCH, "lr": LR},
          open("runs/metrics.json", "w", encoding="utf-8"), indent=2)
