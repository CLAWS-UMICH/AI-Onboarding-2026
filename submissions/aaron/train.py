"""Step 1: fine-tune MPNet + linear head on train.jsonl.

usage: uv run python train.py [--final]

1. Train 3 seeds on the same 85/15 split as baseline.py, early stopping on val macro-F1.
   Report each seed and mean +- std on val / clean / attack.
2. Retrain once on ALL of train.jsonl for the typical best epoch count -> runs/mpnet_ft/model.pt
--final also scores test_final: only run it once, at the very end.
"""

import argparse
import json
import math
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from common import (
    HERE,
    LABEL2ID,
    LABELS,
    SEEDS,
    confusion_plot,
    load,
    load_test,
    score,
    split,
)
from model import DEVICE, MODEL, IntentModel

EPOCHS = 10  # max; early stopping usually ends sooner
PATIENCE = 3  # stop after this many epochs without a better val macro-F1
BATCH = 32
LR_ENCODER = 2e-5  # small: MPNet is already trained, big steps would wreck it
LR_HEAD = 1e-3  # big: the head starts random
WARMUP = 0.1  # fraction of steps the learning rate ramps up over
SMOOTHING = 0.1  # train toward 90% confidence, so confidences stay honest for the unhandled threshold

OUT = HERE / "runs" / "mpnet_ft"
loss_fn = nn.CrossEntropyLoss(label_smoothing=SMOOTHING)


def set_seed(seed):
    random.seed(seed)  # batch shuffling
    np.random.seed(seed)
    torch.manual_seed(seed)  # head init + dropout (CPU and GPU)


def batches(X, y, shuffle):
    order = list(range(len(X)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), BATCH):
        idx = order[i : i + BATCH]
        yield (
            [X[j] for j in idx],
            torch.tensor([LABEL2ID[y[j]] for j in idx], device=DEVICE),
        )


@torch.no_grad()
def evaluate(model, X, y):
    """-> (average loss, predicted labels). eval() turns dropout off."""
    model.eval()
    total, preds = 0.0, []
    for xb, yb in batches(X, y, shuffle=False):
        logits = model(xb)
        total += loss_fn(logits, yb).item() * len(xb)
        preds += logits.argmax(dim=1).tolist()
    return total / len(X), [LABELS[p] for p in preds]


def train(Xtr, ytr, seed, epochs, Xval=None, yval=None):
    """Train one model. With val data: keep the best epoch and stop early. -> (model, history, best_epoch)"""
    set_seed(seed)
    model = IntentModel().to(DEVICE)
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": LR_ENCODER},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ]
    )
    steps = epochs * math.ceil(len(Xtr) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)

    history, best_f1, best_epoch, best_state, bad = [], -1.0, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct = 0.0, 0
        for xb, yb in batches(Xtr, ytr, shuffle=True):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()  # compute how each weight should change
            opt.step()  # change them
            sched.step()  # move the learning rate along warmup/decay
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        row = {
            "epoch": epoch,
            "train_loss": total / len(Xtr),
            "train_acc": correct / len(Xtr),
        }

        if Xval is not None:
            val_loss, val_pred = evaluate(model, Xval, yval)
            val_f1 = f1_score(
                yval, val_pred, labels=LABELS, average="macro", zero_division=0
            )
            row |= {
                "val_loss": val_loss,
                "val_acc": accuracy_score(yval, val_pred),
                "val_macro_f1": val_f1,
            }
            if val_f1 > best_f1:
                best_f1, best_epoch, bad = val_f1, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                bad += 1
        history.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if Xval is not None and bad >= PATIENCE:
            print(f"early stop: no val improvement for {PATIENCE} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def plot_curves(history, path, title):
    """Loss and accuracy per epoch. Train falling while val rises = overfitting."""
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    a.plot(ep, [h["val_loss"] for h in history], "o-", label="val")
    a.set(title="loss", xlabel="epoch")
    b.plot(ep, [h["train_acc"] for h in history], "o-", label="train")
    b.plot(ep, [h["val_acc"] for h in history], "o-", label="val")
    b.set(title="accuracy", xlabel="epoch")
    for ax in (a, b):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--final", action="store_true")
args = ap.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
texts, labels = load(HERE / "data" / "train.jsonl")
Xtr, Xval, ytr, yval = split(texts, labels)  # same split as baseline.py
tests = {
    name: load_test(name)
    for name in ["clean", "attack"] + (["final"] if args.final else [])
}
print(f"device {DEVICE}, train {len(Xtr)} / val {len(Xval)}")

# 1. Three seeds on the fixed split: how good is it, and how much does luck move it?
runs = []
for seed in SEEDS:
    print(f"\n######## seed {seed}")
    model, history, best_epoch = train(Xtr, ytr, seed, EPOCHS, Xval, yval)
    plot_curves(
        history, OUT / f"curves_seed{seed}.png", f"MPNet fine-tune, seed {seed}"
    )
    run = {"seed": seed, "best_epoch": best_epoch, "history": history}
    for name, (x, y) in {"val": (Xval, yval), **tests}.items():
        _, pred = evaluate(model, x, y)
        run[name] = score(x, y, pred, title=f"seed {seed} {name}")
        if name == "attack" and seed == SEEDS[0]:
            confusion_plot(
                y, pred, OUT / "confusion_attack.png", f"attack, seed {seed}"
            )
    runs.append(run)
    del model
    torch.cuda.empty_cache()

summary = {}
for name in ["val", *tests]:
    for m in ["accuracy", "macro_f1"]:
        vals = [r[name][m] for r in runs]
        summary[f"{name}_{m}"] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4),
        }

# 2. Final model: all data (train + val), typical best epoch count, no early stopping (no val left).
final_epochs = round(statistics.median(r["best_epoch"] for r in runs))
print(f"\n######## final model: all {len(texts)} rows, {final_epochs} epochs")
model, _, _ = train(texts, labels, SEEDS[0], final_epochs)
torch.save(model.state_dict(), OUT / "model.pt")
final_scores = {}
for name, (x, y) in tests.items():
    _, pred = evaluate(model, x, y)
    final_scores[name] = score(x, y, pred, title=f"final model {name}")

hyper = {
    "epochs_max": EPOCHS,
    "patience": PATIENCE,
    "batch": BATCH,
    "lr_encoder": LR_ENCODER,
    "lr_head": LR_HEAD,
    "warmup": WARMUP,
    "label_smoothing": SMOOTHING,
    "seeds": SEEDS,
}
metrics = {
    "model": MODEL,
    "method": "fine-tune encoder + linear head",
    # guide's required fields: held-out (val) score, mean of 3 seeds
    "accuracy": summary["val_accuracy"]["mean"],
    "macro_f1": summary["val_macro_f1"]["mean"],
    "summary": summary,
    "final_model": {"epochs": final_epochs, "trained_on": len(texts), **final_scores},
    "runs": [{k: v for k, v in r.items() if k != "history"} for r in runs],
    "hyperparams": hyper,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
(OUT / "history.json").write_text(
    json.dumps({r["seed"]: r["history"] for r in runs}, indent=1), encoding="utf-8"
)
(HERE / "runs" / "metrics.json").write_text(
    json.dumps(metrics, indent=1), encoding="utf-8"
)
(HERE / "runs" / "label2id.json").write_text(
    json.dumps(LABEL2ID, indent=1), encoding="utf-8"
)

print("\n######## summary (mean +- std over seeds)")
for k, v in summary.items():
    print(f"  {k}: {v['mean']:.3f} +- {v['std']:.3f}")
print(f"  final model ({final_epochs} epochs, all data): {final_scores}")
