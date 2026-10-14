"""Classify a sentence with the fine-tuned model.

usage:
  uv run python predict.py "add a waypoint named Olympus Ridge"   # one sentence -> JSON
  uv run python predict.py                                        # interactive loop, Ctrl-C to quit
  uv run python predict.py --check                                # self-check of the rule-based parts (no model)
"""
import json
import re
import sys

import torch

from common import HERE, LABELS

THRESHOLD = 0.5    # top confidence below this -> unhandled (to be tuned on val later)
PAIR_MARGIN = 0.15  # top-2 are a known pair and this close -> report both
TOP_K = 3
MODEL_PATH = HERE / "runs" / "mpnet_ft" / "model.pt"

# Sibling intents that differ by one word. If the model can't decide between them, say so.
PAIRS = [
    {"get_oxy_pri_storage", "get_oxy_sec_storage"},
    {"get_oxy_pri_pressure", "get_oxy_sec_pressure"},
    {"get_fan_pri_rpm", "get_fan_sec_rpm"},
    {"get_scrubber_a_co2_storage", "get_scrubber_b_co2_storage"},
    {"get_coolant_gas_pressure", "get_coolant_liquid_pressure"},
    {"get_primary_battery_level", "get_secondary_battery_level"},
]
NAME_INTENTS = {"Get_coordinates", "Set_navigation_target", "Add_waypoint", "Delete_waypoint", "Complete_task"}

# ponytail: stop-word slot extraction. Names containing these words lose them ("the lander" -> "lander").
# Upgrade path: a token-tagging model, if the name test shows this failing.
SLOT_STOP = set("""
a an the uh um er please can you could just now quick real quickly hey okay ok copy over ev1 ev2 houston
add drop put place set make create save mark new pin marker waypoint waypoints point here named called call it as at
take me to navigate nav navigation go head bring route target destination toward towards
where is whats what's what are coordinates coords coordinate location position of for find get give show tell
delete remove clear erase rid kill
task complete completed done finished finish with i i'm im we we're have has that this off check
""".split())

SPLIT_RE = re.compile(r"\b(?:and then|and also|and|then|also)\b")


def extract_slot(text):
    """Longest run of words that aren't command/filler words, original casing kept."""
    words = re.findall(r"[\w'-]+", text)
    runs, cur = [], []
    for w in words + ["the"]:  # sentinel flushes the last run
        if w.lower() in SLOT_STOP:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(w)
    return " ".join(max(runs, key=len)) if runs else None


class Predictor:
    def __init__(self, path=MODEL_PATH):
        from model import DEVICE, IntentModel  # loads torch models; kept out of --check
        self.model = IntentModel().to(DEVICE)
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()

    @torch.no_grad()
    def probs(self, texts):
        return torch.softmax(self.model(texts), dim=1).cpu()

    def classify(self, text):
        """One clause -> (intent, confidence, top_k, maybe_pair)."""
        p = self.probs([text])[0]
        conf, idx = p.topk(TOP_K)
        top = [(LABELS[i], round(c.item(), 3)) for c, i in zip(conf, idx)]
        (i1, c1), (i2, c2) = top[0], top[1]
        pair = {i1, i2} in PAIRS and c1 - c2 < PAIR_MARGIN
        if c1 < THRESHOLD and not (pair and c1 + c2 >= THRESHOLD):
            return "unhandled", c1, top, None
        return i1, c1, top, (i2, c2) if pair else None

    def predict(self, text):
        intent, conf, top, pair = self.classify(text)
        out = {"text": text, "selected_intent": intent, "confidence": conf}
        if intent in NAME_INTENTS:
            out["slot"] = extract_slot(text)
        if pair:  # "scrubber CO2": A or B, show both
            out["or"] = {"intent": pair[0], "confidence": pair[1]}
        out["top_k"] = top

        # Two commands? Keep the split only if every part is a confident, real, different intent.
        parts = [p.strip() for p in SPLIT_RE.split(text) if p.strip()]
        if len(parts) > 1:
            results = [self.classify(p) for p in parts]
            intents = [r[0] for r in results]
            if "unhandled" not in intents and len(set(intents)) == len(intents):
                first = results[0]
                out |= {"selected_intent": first[0], "confidence": first[1], "top_k": first[2]}
                out.pop("or", None)
                out.pop("slot", None)
                if first[0] in NAME_INTENTS:
                    out["slot"] = extract_slot(parts[0])
                out["also"] = [{"intent": r[0], "confidence": r[1], "text": p} for r, p in zip(results[1:], parts[1:])]
        return out


def check():
    cases = {
        "add a waypoint named Olympus Ridge": "Olympus Ridge",
        "mark soil sample complete": "soil sample",
        "take me to Basalt Knoll please": "Basalt Knoll",
        "where is station 4": "station 4",
        "uh delete the delta-9 waypoint": "delta-9",
        "finished the core drill": "core drill",
    }
    for text, want in cases.items():
        got = extract_slot(text)
        assert got == want, f"{text!r}: got {got!r}, want {want!r}"
    assert [p.strip() for p in SPLIT_RE.split("open vitals and check heart rate")] == ["open vitals", "check heart rate"]
    print(f"ok: {len(cases)} slot cases + split")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
        sys.exit()
    predictor = Predictor()
    if len(sys.argv) > 1:
        print(json.dumps(predictor.predict(" ".join(sys.argv[1:]))))
        sys.exit()
    print("type a command, Ctrl-C to quit")
    try:
        while True:
            text = input("> ").strip()
            if text:
                print(json.dumps(predictor.predict(text), indent=1))
    except (KeyboardInterrupt, EOFError):
        print()
