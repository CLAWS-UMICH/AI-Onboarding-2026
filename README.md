# AI-Onboarding-2026
Onboarding repo for the 2026-27 year, AI subteam.

**Task:** write training data (by hand, synthetically generated, or both) for 36 EVA voice intents (35 commands + `unhandled`), fine-tune a small text model (MiniLM by default, 3 options in `MODELS.md`) to classify them, and test it on sentences it has never seen. Voice (Whisper + TTS) is optional.

Full guide: [`claws_intent_onboarding.pdf`](claws_intent_onboarding.pdf)

## Layout
```
.
├── claws_intent_onboarding.pdf   # the guide (read this first)
├── requirements.txt
├── code/                         # shared; never edit
│   ├── intents.json              # the 36 intent labels
│   ├── validate.py               # check your data
│   └── baseline.py, model.py, train.py, predict.py, voice.py
│                                 # outlines only (comments); write your own
└── submissions/<your-name>/      # your work goes here only
    ├── MODELS.md                 # the 3 encoders + how to switch
    ├── intents.json              # copy of code/intents.json
    ├── data/train.jsonl
    ├── *.py                      # your code, structured however you like
    ├── runs/
    └── notes.md
```

## Quick start
Use Python 3.10–3.12.
```bash
git clone https://github.com/CLAWS-UMICH/AI-Onboarding-2026.git
cd AI-Onboarding-2026
git checkout -b onboarding/<your-name>

python3 -m venv .venv && source .venv/bin/activate    # macOS / Linux
py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
cd submissions/<your-name>

python ../../code/validate.py data/train.jsonl   # until "0 problems" (30+ rows per intent, handwritten or synthetic)
```
Then write your own baseline, training and prediction code in your folder. The files in `code/` are short outlines of what each step should do; how you do it is up to you.

## Submit
Open **one PR** from `onboarding/<your-name>` when everything is done. It must include your code, `data/train.jsonl`, `runs/metrics.json`, `runs/label2id.json`, `baseline.txt` and a filled-in `notes.md`. Only change files in your own folder. `.gitignore` keeps `model.pt` out.

To edit the guide, change the `.tex`, run `latexmk -pdf claws_intent_onboarding.tex`, and commit both files.
