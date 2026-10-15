"""Laya zero-shot: no training, it picks one of the 36 intents from their descriptions.

usage: uv add laya   (once)
       uv run python laya_eval.py
Scores the same val / clean / attack sets as baseline.py and train.py -> runs/laya/metrics.json
"""
import json

import laya

from common import HERE, load, load_test, score, split

CHECKPOINT = "convaiinnovations/laya"  # English ModernBERT-large, 421M
QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "Which command is an astronaut giving their spacesuit heads-up display? "
                        "Pick unhandled if it is not a command.",
        "criteria": {x["intent"]: x["description"]
                     for x in json.loads((HERE / "intents.json").read_text(encoding="utf-8"))},
    }
}

agent = laya.load(CHECKPOINT)
agent.cfg["head_max_len"] = 512  # default 192 tokens fits ~20 options; we have 36

texts, labels = load(HERE / "data" / "train.jsonl")
_, Xval, _, yval = split(texts, labels)  # same val rows as the other models
evals = {"val": (Xval, yval), "clean": load_test("clean"), "attack": load_test("attack")}

metrics = {"model": CHECKPOINT, "method": "zero-shot (no training)"}
for name, (x, y) in evals.items():
    results = agent.predict_batch(x, QUESTION, batch_size=16)  # all at once runs out of GPU memory
    pred = [r["answers"]["intent"]["choice"] for r in results]
    metrics[name] = score(x, y, pred, title=f"laya {name}")

out = HERE / "runs" / "laya"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
print({k: v for k, v in metrics.items() if isinstance(v, dict)})
