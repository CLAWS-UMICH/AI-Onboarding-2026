"""Step 0: frozen sentence embeddings + logistic regression. No fine-tuning.
Prints the report and saves it to baseline.txt."""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model import MODEL_NAME

rows = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8") if line.strip()]
texts = [r["text"] for r in rows]
labels = [r["intents"][0] for r in rows]
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.15, stratify=labels, random_state=0)

encoder = SentenceTransformer(MODEL_NAME)
clf = LogisticRegression(max_iter=2000).fit(encoder.encode(X_tr), y_tr)
report = classification_report(y_te, clf.predict(encoder.encode(X_te)), zero_division=0)
report = f"model: {MODEL_NAME}\n\n{report}"
print(report)
Path("baseline.txt").write_text(report, encoding="utf-8")
