"""Classify sentences with the fine-tuned model and pull out the name for the five name intents.

Usage:
  python predict.py "add a waypoint named Olympus Ridge"
  python predict.py              (then type one sentence per line; Ctrl-D to quit)
"""
import json
import re
import sys

import torch

from common import RUNS, device
from model import IntentModel, load_tokenizer, predict_proba

NAME_INTENTS = {"Get_coordinates", "Set_navigation_target", "Add_waypoint", "Delete_waypoint", "Complete_task"}

# Self-corrections: keep only what comes after the last one ("take me to X, no wait, Y" -> "Y").
REPAIR = re.compile(r"\b(?:no wait|wait no|no actually|actually no|actually|sorry|i mean|i meant|scratch that|wait)\b,?\s*", re.I)

# A waypoint/marker/pin noun that is part of a numbered name ("waypoint 12") is not a stop word.
NOUN = r"(?:waypoint|marker|pin)s?\b(?!\s*\d)"

# Words that end a name: clause breaks, politeness, trailing context.
STOP = (r"(?=\s*(?:[,?.!:;]|\s[-–—]|$|\s(?:so|and|real quick|because|since|before|after|for me|please|pls|thanks|right now|right here|rn|next|"
        r"from here|from my|from me|relative to|on the map|on the grid|on my|off my|off the|from the|at my|at the|at this|"
        r"is|are|located|exactly|there|here|with that name|we dont|we don't|i need|i want|"
        r"as (?:a |my |the )?(?:waypoint|marker|pin|destination|target|nav target|active destination|done|complete|completed|finished)|"
        rf"{NOUN}|task|done|complete|completed|finished|gone|removed|coordinates|location|position|"
        r"to (?:my|the) waypoints?|to complete)\b))")

NAMED = r"named|called|labell?ed|call it|make it"
# Cues tried in order; the first one that yields a real name wins.
CUES = {
    "Get_coordinates": [r"where(?:'s| is| exactly is| did we (?:leave|put)| on the grid is| would i find)?",
                        r"coord(?:inate)?s (?:of|for)", r"(?:location|position|grid ref(?:erence)?) (?:of|for|check on)",
                        r"lat long for", r"how far (?:away )?is(?: it to)?", r"how far am i from", r"distance to"],
    "Set_navigation_target": [r"(?:take me|navigate(?: me)?|guide me(?: over)?|head(?:ing)?(?: over)?|route me(?: back)?|routed|"
                              r"route|get me(?: over)?|lead(?: me| the way)?(?: back)?|walk me|plot a course|guidance|go|"
                              r"directions|start navigation|back)(?:\s+(?:to|toward|towards|for))",
                              r"set(?: my route to| nav target to)?", r"destination", r"next stop is"],
    "Add_waypoint": [NAMED, r"as", NOUN.replace(r"\b(?!\s*\d)", r"\b(?!\s*\d)(?: for)?"),
                     r"(?:add|drop|save|create|place|mark|pin)", r"for"],
    "Delete_waypoint": [r"(?:delete|remove|clear|erase|get rid of|wipe|take)", r"(?:waypoint|marker|pin) for"],
    "Complete_task": [r"(?:check off|tick off|mark|log|complete task|completed(?: the)?|finished(?: the| with the)?|"
                      r"wrapped up(?: the)?|done with)"],
}
# "the X waypoint" / "the X task" works for every name intent.
THE_X_NOUN = re.compile(r"\bthe\s+(?P<name>.+?)\s+(?:waypoint|marker|pin|task)\b", re.I)
PRONOUNS = {"that", "this", "it", "its", "there", "here", "them", "one"}
# A "name" made only of these words is not a name ("where am I" -> None, "mark this location" -> None).
NOT_A_NAME = PRONOUNS | {"am", "i", "me", "we", "us", "you", "location", "spot", "position", "place", "coords",
                         "coordinates", "real", "quick", "now", "please", "right", "again", "instead",
                         "done", "finished", "complete", "completed"}
# Place names never start with a command verb (task names can: "check battery terminals").
VERB_START = {"mark", "add", "drop", "save", "create", "place", "pin", "delete", "remove", "clear", "erase", "take",
              "navigate", "guide", "route", "set", "send", "go", "get", "where", "what", "coords", "show", "find"}
FILLERS = (r"uh|um|okay|ok|alright|so|hey|eva|real quick|thanks|now|lets|let's|the|a|an|my|our|new|"
           r"this(?: spot| location)?|here|to|as|for|from|destination|named|called|labell?ed")
LEADING_JUNK = re.compile(rf"^(?:(?:{FILLERS}|{NOUN}|task\b)\b[\s,]*)+", re.I)
LEADING_FILLER = re.compile(rf"^(?:(?:{FILLERS}|task)\b[\s,]*)+", re.I)  # task names keep words like "waypoint"
TRAILING_JUNK = re.compile(r"(?:\s+(?:waypoint|marker|pin|task|it|that|is|as))+$", re.I)
CAPITALIZED = re.compile(r"\b(?!I\b|EVA\b)[A-Z][\w']*(?:\s+(?:[A-Z][\w']*|\d+))*")


def clean(name, intent):
    junk = LEADING_FILLER if intent == "Complete_task" else LEADING_JUNK
    name = junk.sub("", name.strip(" ,?.!:;-–—")).strip()
    name = TRAILING_JUNK.sub("", name).strip(" ,?.!:;-–—")
    name = re.sub(r"'s$", "", name)
    words = name.lower().split()
    if not words or words[0] in PRONOUNS or all(w in NOT_A_NAME for w in words):
        return None
    if intent != "Complete_task" and words[0] in VERB_START:
        return None
    return name


def extract_slot(text, intent):
    tail = REPAIR.split(text)[-1].strip()
    for part in ([tail, text] if tail != text else [text]):
        for cue in [NAMED] + CUES[intent]:
            for m in re.finditer(rf"\b(?:{cue})\s+(?P<name>.+?){STOP}", part, re.I):
                if name := clean(m["name"], intent):
                    return name
        if (m := THE_X_NOUN.search(part)) and (name := clean(m["name"], intent)):
            return name
        # Name first: "Crater Rim, where is that" / "soil sample is done" / "Ridge Point waypoint, add it".
        m = re.match(rf"^(?P<name>.+?){STOP}", part, re.I)
        if m and not re.match(r"^(?:i|im|i'm|we|we're|can|could|would|will|is|hey|please|what|whats|what's)\b", part, re.I):
            if name := clean(m["name"], intent):
                return name
        if m := CAPITALIZED.search(part):
            if name := clean(m.group(), intent):
                return name
    return None  # no name found: the command is missing its name


class Predictor:
    def __init__(self):
        ckpt = torch.load(RUNS / "model.pt", map_location="cpu")
        self.labels = list(json.loads((RUNS / "label2id.json").read_text(encoding="utf-8")))
        self.threshold = json.loads((RUNS / "metrics.json").read_text(encoding="utf-8"))["unhandled_threshold"]
        self.dev = device()
        self.tokenizer = load_tokenizer(ckpt["model"])
        self.model = IntentModel(ckpt["model"], len(self.labels))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.dev).eval()

    def __call__(self, text):
        probs = predict_proba(self.model, self.tokenizer, [text], self.dev)[0]
        conf, idx = probs.max(0)
        intent, conf = self.labels[idx], round(conf.item(), 3)
        out = {"text": text, "selected_intent": intent, "confidence": conf}
        if conf < self.threshold:
            out.update(selected_intent="unhandled", best_guess=intent)
        elif intent in NAME_INTENTS:
            out["slot"] = extract_slot(text, intent)
        return out


def main():
    predictor = Predictor()
    sentences = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else (line.strip() for line in sys.stdin)
    for s in sentences:
        if s:
            print(json.dumps(predictor(s)), flush=True)


if __name__ == "__main__":
    main()
