"""Step 0 baseline: frozen sentence embeddings + logistic regression. The number to beat.

Usage: python baseline.py          (MODEL=... to change encoder)
"""
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import HERE, LABELS, MODEL, load_rows, split


def main():
    train, val, test = split(load_rows())
    encoder = SentenceTransformer(MODEL)
    embed = lambda rows: encoder.encode([t for t, _ in rows], normalize_embeddings=True, batch_size=128)

    # Fit on train only, same as the fine-tune, so the comparison is like with like.
    clf = LogisticRegression(max_iter=5000)
    clf.fit(embed(train), [y for _, y in train])

    y_true = [y for _, y in test]
    y_pred = clf.predict(embed(test))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=range(len(LABELS)),
                                   target_names=LABELS, digits=3, zero_division=0)

    text = (f"model: {MODEL}\n"
            f"method: frozen embeddings + logistic regression\n"
            f"split: train={len(train)} val={len(val)} test={len(test)} (evaluated on test)\n"
            f"accuracy: {acc:.4f}\nmacro_f1: {macro_f1:.4f}\n\n{report}")
    (HERE / "baseline.txt").write_text(text, encoding="utf-8")
    print(text)

    misses = [(t, LABELS[y], LABELS[p]) for (t, y), p in zip(test, y_pred) if y != p]
    print(f"{len(misses)} misses:")
    for t, y, p in misses:
        print(f"  {y} -> {p}: {t!r}")


if __name__ == "__main__":
    main()
