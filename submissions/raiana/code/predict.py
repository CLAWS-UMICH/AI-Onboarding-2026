"""
DIRECTIONS -------------------------------------------------------

Classify a sentence. Write your own version in your folder.

Outline (do it however you like):
- Load your saved model and label map from runs/.
- Take a sentence from the command line, print JSON with the text,
  selected_intent and confidence.
- Low confidence -> answer "unhandled" instead of guessing.
- For the five name intents (Get_coordinates, Set_navigation_target, Add_waypoint,
  Delete_waypoint, Complete_task) also pull out the name as "slot".

STEPS -----------------------------------------------------------

1. Read runs/metrics.json first and set INTENT_MODEL from it, BEFORE importing model.py,
   so we load the encoder train.py actually used (not whatever the terminal has set now)
2. Load runs/label2id.json + runs/model.pt (on CPU, fast enough for one sentence)
3. Softmax the logits; if the top probability is under MIN_CONF, answer "unhandled"
4. For the five name intents, extract the name as "slot":
   - if "named"/"called" appears, the name is what follows it
   - otherwise peel command words off both ends
     ("mark the egress task complete" -> "egress", "erase the pebble field waypoint" -> "pebble field")
5. Usage:
     python code/predict.py "add a waypoint named Olympus Ridge"   one sentence
     python code/predict.py                                         type sentences, blank line quits
"""

# LIBRARIES ------------------------------------------------------

import json
import os
import re
import sys
from pathlib import Path

import torch # noqa: E402
from model import IntentModel, load_tokenizer, tokenize # noqa: E402

# VARIABLES ------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent # submissions/raiana-new/
RUNS = ROOT / "runs"

if not (RUNS / "metrics.json").exists():
    sys.exit("No runs/metrics.json yet. Run python code/train.py first.")
METRICS = json.loads((RUNS / "metrics.json").read_text(encoding="utf-8"))
os.environ["INTENT_MODEL"] = METRICS["model"] # must happen before importing model.py

MIN_CONF = 0.5 # below this, answer "unhandled" instead of guessing

SLOT_INTENTS = {"Get_coordinates", "Set_navigation_target", "Add_waypoint", "Delete_waypoint", "Complete_task"}

# Words that can surround a name but are never part of it
FILLER = {"um", "uh", "hey", "please", "now", "thanks", "ok", "okay", "so", "and"}
LEAD = FILLER | {
    # asking / politeness
    "can", "could", "would", "you", "i", "want", "need", "lets", "let's", "let", "us", "me", "my",
    "what", "whats", "what's", "where", "where's", "is", "are", "show", "give", "tell", "find", "get",
    # actions
    "set", "add", "create", "drop", "make", "put", "place", "new", "save", "pin",
    "delete", "remove", "clear", "erase", "rid", "mark", "complete", "check", "off", "finish", "finished",
    "navigate", "route", "take", "go", "guide", "head", "bring", "start",
    # objects
    "waypoint", "marker", "point", "task", "navigation", "nav", "target",
    "position", "location", "coordinates", "coords", "destination",
    # glue
    "the", "a", "an", "to", "for", "of", "at", "towards", "toward", "as",
}
TRAIL = FILLER | {
    "waypoint", "marker", "pin", "point", "task", "location", "position", "coordinates", "coords",
    "complete", "completed", "done", "finished", "as", "is", "please",
}
NAME_CUE = re.compile(r"\b(?:named|called)\b", re.I)

# HELPER FUNCTIONS ------------------------------------------------

# Goal: strip punctuation for comparing words against the sets above
def _bare(word):
    return re.sub(r"[^\w']", "", word).lower()


# Goal: extract a person's name from text by identifying portion after a name-cue pattern
# Example of name-cue pattern: "My name is..."
def extract_slot(text):

    # Searches fora ll occurences of name-cue pattern, stores all matches as a list
    cue = list(NAME_CUE.finditer(text))

    # If a cue was found, take everything after last cue match and split into words, otherwise use entire text
    # This isolates likely name region
    words = (text[cue[-1].end():] if cue else text).split()

    # If no cue pattern found, meaning we're using whole text
    if not cue:

        # Remove trailing words that are common filler (from TRAIL set)
        while words and _bare(words[0]) in LEAD:
            words.pop(0)

    # Remove trailing words that are common filler (from TRAIL set)
    while words and _bare(words[-1]) in TRAIL:
        words.pop()

    # Remove articles
    while words and _bare(words[0]) in {"the", "a", "an"}:
        words.pop(0)

    # Obtain name
    name = " ".join(words).strip(" .,!?'\"")

    # Return name or none (if none found)
    return name or None

# Goal: Predictor class - loads trained intent classification model to predict intents from text with confidence scores
class Predictor:
    def __init__(self):
        # Loads saved intent name --> ID mapping from JSON file
        label2id = json.loads((RUNS / "label2id.json").read_text(encoding="utf-8"))

        # Creates list where index matches model's output ID - sorts intent names by their ID value so self.labels[0] is the intent with ID 0
        self.labels = sorted(label2id, key=label2id.get)

        # Loads same tokenizer during training
        self.tokenizer = load_tokenizer()

        # Creates new model instance with correct number of output classes
        self.model = IntentModel(len(self.labels))

        # Loads trained weights from saved file onto CPU
        self.model.load_state_dict(torch.load(RUNS / "model.pt", map_location="cpu"))

        # Switches model to eval mode (disables dropout)
        self.model.eval() # turns dropout off

    # Goal: Defines method that runs when you call the object like a function
    def __call__(self, text):
        # Disables gradient computation for faster inference
        with torch.no_grad():
            probs = self.model(tokenize(self.tokenizer, [text], "cpu")).softmax(dim=1)[0]

        # Finds highest probability (confidence) and corresponding intent class index
        conf, idx = probs.max(dim=0)

        # Converts class index into intent name, if confidence below min threshold, returns "unhandled"
        intent = self.labels[idx.item()] if conf.item() >= MIN_CONF else "unhandled"

        result = {"text": text, "selected_intent": intent, "confidence": round(conf.item(), 3)}

        # If predicted intent is one that requires slot extraction (like a name), add extracted slot value to result
        if intent in SLOT_INTENTS:
            result["slot"] = extract_slot(text)

        return result


_predictor = None

# Load model (predictor) 
def predict(text):
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor(text)


# MAIN LOOP ---------------------------------------------------------

if __name__ == "__main__":
    # If command line arguments were provided, join all arguments into single text string, predict intent, convert result to JSON, print it
    if len(sys.argv) > 1:
        print(json.dumps(predict(" ".join(sys.argv[1:]))))

    # If no arguments, if blank line, break
    else:
        print(f"model: {METRICS['model']}. Type a sentence, blank line to quit.")
        for line in sys.stdin:
            if not line.strip():
                break
            print(json.dumps(predict(line.strip())))
