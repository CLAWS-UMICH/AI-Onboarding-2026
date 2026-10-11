"""
DIRECTIONS -------------------------------------------------------

Step 0: baseline. No fine-tuning. Write your own version in your folder.

Outline (do it however you like):
- Load data/train.jsonl, hold out ~15% (stratified, fixed seed).
- Embed sentences with a frozen encoder (sentence-transformers is installed).
- Fit a simple classifier (e.g. sklearn logistic regression) on the embeddings.
- Print a per-intent report and save it to baseline.txt (include the model name).

STEPS -----------------------------------------------------------

1. Frozen encoder embeddings + logistic regression, no fine-tuning
2. Embeds every sentence once with the pretrained encoder, fits linear classifier on those vectors
3. Fits on the same 85% of train.jsonl that train.py trains on (the other 15% is train.py's validation split)
4. Scores it on data/test.jsonl, a separate handwritten file that never comes from train.jsonl,
   writes report to baseline.txt. That number train.py has to beat
5. Picking default MiniLM for now
"""

# LIBRARIES ------------------------------------------------------

import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# PATH RESOLVING -------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # submissions/raiana/
DATA = ROOT / "data" / "train.jsonl"
TEST_DATA = ROOT / "data" / "test.jsonl" # handwritten, never split from train.jsonl, and used for final score because accuracy was coming up WAY too high and me need to check
OUT = ROOT / "baseline.txt"

MODEL_NAME = os.environ.get("INTENT_MODEL", "sentence-transformers/all-MiniLM-L6-v2") # default is MiniLM
VAL_SIZE = 0.15 # validation split carved from train.jsonl (train.py watches it each epoch)
SEED = 56 # makes the random split reproducible

# Note: train.py must use same VAL_SIZE and SEED so both train on the same 85%, then both get scored on test.jsonl

# HELPER FUNCTIONS ------------------------------------------------

# Returns (texts, labels) from a JSON file
# One {"text", "intents": [label]} per line
def load_rows(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f: # utf-8 also default used by VS Code
        for line in f:
            # removes all whitespace and returns result as string, this being true = more data exists
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                # intents is a list with one element (i.e. the intent)
                labels.append(row["intents"][0])
    return texts, labels

# Returns (texts, labels) from test.jsonl, quits if it's missing or shares sentences with train.jsonl
def load_test(train_texts):
    if not TEST_DATA.exists():
        sys.exit(f"No {TEST_DATA.relative_to(ROOT)} yet. Write it by hand, same format as train.jsonl.")
    texts, labels = load_rows(TEST_DATA)

    # a test sentence that's also in train.jsonl should be flagged
    seen = {t.strip().lower() for t in train_texts}
    overlap = [t for t in texts if t.strip().lower() in seen]
    if overlap:
        sys.exit("These test.jsonl sentences are also in train.jsonl, remove one copy:\n  " + "\n  ".join(overlap))
    return texts, labels

# MAIN LOOP ---------------------------------------------------------

def main():
    texts, labels = load_rows(DATA)

    # only the 85% train part is used here and ze 15% val part is for train.py's per-epoch checks
    train_x, _, train_y, _ = train_test_split(
        texts, labels, test_size=VAL_SIZE, stratify=labels, random_state=SEED
    )
    test_x, test_y = load_test(texts)
    print(f"{MODEL_NAME}: {len(train_x)} train / {len(test_x)} test sentences (test from {TEST_DATA.name})")

    # Loads (downloads weights of) pre-trained model from Hugging Face and sets it up so you can call .encode() on sentences (which outputs vectors)
    encoder = SentenceTransformer(MODEL_NAME)

    # turn sentences into vectors
    # train_vecs encodes 956 training sentences (return shape: (956, 384)) and test_vecs encodes 169 test sentences (return shape: (169, 384))
        # 384 = hidden size of MiniLM (i.e. it is how many numbers each sentence gets encoded into)
    # batch_size = 64 means it processes 64 sentences at one time (faster)
    train_vecs = encoder.encode(train_x, batch_size=64, show_progress_bar=False)
    test_vecs = encoder.encode(test_x, batch_size=64, show_progress_bar=False)

    # creates the classifier, max_iter = 2000 is how many gradient descent steps it is allowed to take while learning
    clf = LogisticRegression(max_iter=2000)

    # Trains the model - it learns which directions in the 384-dimensional space point toward each intent
    clf.fit(train_vecs, train_y)

    # Runs trained classifier on 169 test vectors and gets back 169 predicted intent labels stored in pred
    pred = clf.predict(test_vecs)

    # Compares pred (predicted labels) and test_y (ground truth labels) to compute accuracy, f1-score, etc.
    # average="macro" means it computes F1 for each intent separately, then averages them equally
        # Used because if one intent has way more test examples (is a "common intent"), that intent would be predicted way more often so it's right more often and simple average would show higher result for accuracy than it should be
            # would not flag the issue that we're just predicting same, COMMON label for 90% intents so we're getting 90% accuracy
        # Ex: 100% on 30 common intents but 0% on 6 rare ones would get this F1 to drop
    acc = accuracy_score(test_y, pred)
    macro_f1 = f1_score(test_y, pred, average="macro")

    # Prints out report
    report = (
        f"model: {MODEL_NAME}\n"
        f"accuracy: {acc:.3f}  macro-F1: {macro_f1:.3f}  "
        f"(train {len(train_x)}, test {len(test_x)} from {TEST_DATA.name})\n\n"
        + classification_report(test_y, pred, zero_division=0)
    )
    print(report)
    OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
