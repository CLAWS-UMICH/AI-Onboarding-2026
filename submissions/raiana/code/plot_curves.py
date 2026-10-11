"""
STEPS -----------------------------------------------------------

*Made by Claude so ik what to do*

1. Read runs/history.json (written by train.py, one entry per epoch)
2. Left plot: train loss vs validation loss per epoch
   - validation loss rising while train loss keeps falling = overfitting, use fewer epochs
   - both still falling at the last epoch = could train longer
3. Right plot: validation accuracy + macro-F1 per epoch, with ze baseline's test accuracy as a dashed line
   (rough reference only: the curves are on the validation split, the baseline is on test.jsonl)
4. Saves runs/learning_curves.png
5. Usage:
     python code/train.py          writes runs/history.json
     python code/plot_curves.py    draws runs/learning_curves.png
"""

# LIBRARIES ------------------------------------------------------

import json
import re
import sys

import matplotlib.pyplot as plt

from baseline import ROOT, OUT as BASELINE_TXT

# VARIABLES ------------------------------------------------------

RUNS = ROOT / "runs"
HISTORY = RUNS / "history.json"
PLOT = RUNS / "learning_curves.png"

# HELPER FUNCTIONS ------------------------------------------------

# Goal: pull the accuracy number out of baseline.txt ("accuracy: 0.953  macro-F1: ..."), None if missing
def baseline_accuracy():
    if not BASELINE_TXT.exists():
        return None
    match = re.search(r"accuracy:\s*([\d.]+)", BASELINE_TXT.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None

# MAIN LOOP ---------------------------------------------------------

def main():
    if not HISTORY.exists():
        sys.exit("No runs/history.json yet. Run python code/train.py first.")
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]

    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(11, 4))

    # Loss curves: ze gap between these two is what shows overfitting
    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    loss_ax.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    loss_ax.legend()

    # Score curves on the validation split
    score_ax.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="val accuracy")
    score_ax.plot(epochs, [row["val_macro_f1"] for row in history], marker="o", label="val macro-F1")
    base = baseline_accuracy()
    if base is not None:
        score_ax.axhline(base, linestyle="--", color="gray", label=f"baseline test acc ({base:.3f})")
    score_ax.set(xlabel="epoch", ylabel="score", title="Validation score")
    score_ax.legend()

    for ax in (loss_ax, score_ax):
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"saved {PLOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
