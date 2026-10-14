"""Shared data loading, split and scoring, so every experiment is graded the same way."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save PNGs without needing a display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
SEEDS = [42, 43, 44]
VAL_FRACTION = 0.15
TEST_SETS = ["clean", "attack", "final"]

# Label ids come from intents.json order, never from the data, so they never change between runs.
LABELS = [
    x["intent"] for x in json.loads((HERE / "intents.json").read_text(encoding="utf-8"))
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load(path):
    """Read a JSONL file -> (texts, labels)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [r["text"] for r in rows], [r["intents"][0] for r in rows]


def split(texts, labels, seed=SEEDS[0]):
    """Stratified train/val split: every intent keeps the same share in both halves."""
    return train_test_split(
        texts, labels, test_size=VAL_FRACTION, stratify=labels, random_state=seed
    )  # Xtr, Xval, ytr, yval


def load_test(name):
    """name in TEST_SETS -> (texts, labels). Never train or tune on 'final'."""
    return load(HERE / "data" / f"test_{name}.jsonl")


def score(texts, true, pred, title=""):
    """Print accuracy, macro-F1, per-intent report and every miss. Returns the two numbers."""
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, labels=LABELS, average="macro", zero_division=0)
    print(f"\n== {title}: accuracy {acc:.3f}, macro-F1 {macro_f1:.3f}, n={len(true)}")
    print(classification_report(true, pred, labels=LABELS, zero_division=0))
    misses = [(t, y, p) for t, y, p in zip(texts, true, pred) if y != p]
    print(f"misses ({len(misses)}):")
    for t, y, p in misses:
        print(f"  {t!r}: true {y}, got {p}")
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4)}


def confusion_plot(true, pred, path, title=""):
    """Save a true-vs-predicted grid as PNG; off-diagonal cells are confusions."""
    fig, ax = plt.subplots(figsize=(16, 16))
    ConfusionMatrixDisplay.from_predictions(
        true, pred, labels=LABELS, ax=ax, xticks_rotation="vertical", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # Self-check: split is stratified and repeatable, scoring math is right.
    texts, labels = load(HERE / "data" / "train.jsonl")
    a, b = split(texts, labels), split(texts, labels)
    assert a[1] == b[1], "same seed must give the same split"
    assert set(a[3]) == set(labels), "every intent must appear in val"
    assert (
        score(["x", "y"], ["unhandled", "close_menu"], ["unhandled", "unhandled"])[
            "accuracy"
        ]
        == 0.5
    )
    print(f"\nok: {len(a[0])} train / {len(a[1])} val, {len(LABELS)} labels")
