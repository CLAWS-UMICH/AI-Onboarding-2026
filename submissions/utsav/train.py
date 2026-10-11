"""Step 1 fine-tune: encoder + head with cross-entropy, early stopping on val, several seeds.

Usage: python train.py [--seeds 42 43 44] [--epochs 15] ...   (MODEL=... to change encoder)

Decisions (early stopping, which seed's model to keep, the unhandled threshold) use val only.
Test is evaluated after those decisions and is what goes into runs/metrics.json.
"""
import argparse
import copy
import json
import math
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import get_linear_schedule_with_warmup

from common import LABEL2ID, LABELS, MODEL, RUNS, device, load_rows, set_seed, split
from model import IntentModel, encode, load_tokenizer, predict_proba


def scores(model, tokenizer, rows, dev):
    probs = predict_proba(model, tokenizer, [t for t, _ in rows], dev)
    y_true = [y for _, y in rows]
    y_pred = probs.argmax(1).tolist()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro"), probs


def train_one(seed, train, val, tokenizer, dev, args):
    set_seed(seed)
    model = IntentModel(MODEL, len(LABELS)).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    val_loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(train), generator=gen).split(args.batch_size):
            batch = [train[i] for i in idx]
            ids, mask = encode(tokenizer, [t for t, _ in batch], dev)
            labels = torch.tensor([y for _, y in batch], device=dev)
            loss = loss_fn(model(ids, mask), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)

        val_acc, val_f1, val_probs = scores(model, tokenizer, val, dev)
        val_loss = val_loss_fn(val_probs.clamp_min(1e-9).log(), torch.tensor([y for _, y in val])).item()
        print(f"  seed {seed} epoch {epoch}: train_loss={total / len(train):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch, "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
        elif epoch - best["epoch"] >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} epochs")
            break

    model.load_state_dict(best["state"])
    return model, best


def pick_threshold(probs, val):
    """Highest confidence cutoff (below it -> unhandled) that costs at most 1 point of val accuracy."""
    y_true = np.array([y for _, y in val])
    conf, pred = probs.max(1)
    conf, pred = conf.numpy(), pred.numpy()
    unhandled = LABEL2ID["unhandled"]
    accs = {}
    for t in np.round(np.arange(0.0, 0.95, 0.05), 2):
        accs[float(t)] = float((np.where(conf < t, unhandled, pred) == y_true).mean())
    best = max(accs.values())
    return max(t for t, a in accs.items() if a >= best - 0.01), accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--smoothing", type=float, default=0.1)
    args = p.parse_args()

    dev = device()
    train, val, test = split(load_rows())
    tokenizer = load_tokenizer(MODEL)
    print(f"model={MODEL} device={dev} train={len(train)} val={len(val)} test={len(test)}")

    runs, best_run = [], None
    for seed in args.seeds:
        start = time.time()
        model, best = train_one(seed, train, val, tokenizer, dev, args)
        test_acc, test_f1, test_probs = scores(model, tokenizer, test, dev)
        run = {"seed": seed, "best_epoch": best["epoch"], "val_macro_f1": round(best["f1"], 4),
               "test_accuracy": round(test_acc, 4), "test_macro_f1": round(test_f1, 4),
               "seconds": round(time.time() - start)}
        print(f"seed {seed}: best epoch {best['epoch']}, val_macro_f1={best['f1']:.4f}, "
              f"test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}")
        runs.append(run)
        # Keep the model with the best VAL score (never pick by test).
        if best_run is None or best["f1"] > best_run["run"]["val_macro_f1"]:
            best_run = {"run": run, "model": model, "test_probs": test_probs}

    model = best_run["model"]
    _, _, val_probs = scores(model, tokenizer, val, dev)
    threshold, threshold_accs = pick_threshold(val_probs, val)

    test_pred = best_run["test_probs"].argmax(1).tolist()
    misses = [(t, LABELS[y], LABELS[p], best_run["test_probs"][i, p].item())
              for i, ((t, y), p) in enumerate(zip(test, test_pred)) if y != p]
    print(f"\nsaved model: seed {best_run['run']['seed']} | {len(misses)} test misses:")
    for t, y, p_, c in misses:
        print(f"  {y} -> {p_} ({c:.2f}): {t!r}")

    test_accs = [r["test_accuracy"] for r in runs]
    test_f1s = [r["test_macro_f1"] for r in runs]
    metrics = {
        "model": MODEL,
        "accuracy": best_run["run"]["test_accuracy"],
        "macro_f1": best_run["run"]["test_macro_f1"],
        "seed_mean_accuracy": round(float(np.mean(test_accs)), 4),
        "seed_std_accuracy": round(float(np.std(test_accs)), 4),
        "seed_mean_macro_f1": round(float(np.mean(test_f1s)), 4),
        "seed_std_macro_f1": round(float(np.std(test_f1s)), 4),
        "saved_seed": best_run["run"]["seed"],
        "unhandled_threshold": threshold,
        "val_accuracy_by_threshold": threshold_accs,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "hyperparams": vars(args),
        "runs": runs,
        "test_misses": [{"text": t, "true": y, "pred": p_, "confidence": round(c, 3)} for t, y, p_, c in misses],
    }

    RUNS.mkdir(exist_ok=True)
    torch.save({"model": MODEL, "state_dict": model.state_dict()}, RUNS / "model.pt")
    (RUNS / "label2id.json").write_text(json.dumps(LABEL2ID, indent=1), encoding="utf-8")
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"\naccuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
          f"(seed mean acc {metrics['seed_mean_accuracy']} ± {metrics['seed_std_accuracy']}) "
          f"threshold={threshold}\nwrote runs/model.pt, runs/label2id.json, runs/metrics.json")


if __name__ == "__main__":
    main()
