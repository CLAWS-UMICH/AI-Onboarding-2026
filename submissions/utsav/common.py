"""Shared setup: paths, encoder choice, label map and the one fixed train/val/test split."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
DATA = HERE / "data" / "train.jsonl"
RUNS = HERE / "runs"

# Switch encoders with an env var, e.g. MODEL=BAAI/bge-small-en-v1.5 python train.py
MODEL = os.environ.get("MODEL", "sentence-transformers/all-mpnet-base-v2")

SPLIT_SEED = 42      # fixed, so baseline and fine-tune see the exact same split
TEST_FRAC = 0.15     # touched once, for the reported numbers
VAL_FRAC = 0.10      # used for every decision: early stopping, model pick, threshold

LABELS = [x["intent"] for x in json.loads((HERE / "intents.json").read_text(encoding="utf-8"))]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_rows(path=DATA):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [(r["text"], LABEL2ID[r["intents"][0]]) for r in rows]


def split(rows):
    """Stratified 75/10/15 train/val/test split. Returns three lists of (text, label_id)."""
    labels = [y for _, y in rows]
    rest, test = train_test_split(rows, test_size=TEST_FRAC, stratify=labels, random_state=SPLIT_SEED)
    rest_labels = [y for _, y in rest]
    train, val = train_test_split(rest, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                  stratify=rest_labels, random_state=SPLIT_SEED)
    return train, val, test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
