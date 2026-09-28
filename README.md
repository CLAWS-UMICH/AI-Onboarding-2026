# AI-Onboarding-2026
Onboarding repo for the 2026-27 year, AI subteam.

**Task:** write training data for 35 EVA voice intents, fine-tune a small text model (MiniLM) to classify them, and test it on sentences it has never seen. Voice (Whisper + TTS) is optional.

Full guide: [`claws_intent_onboarding.pdf`](claws_intent_onboarding.pdf)

## Layout
```
.
├── claws_intent_onboarding.pdf   # the guide (read this first)
├── code/                         # shared scripts; don't edit or copy
│   ├── intents.json              # the 35 intent labels
│   ├── validate.py               # check your data
│   ├── baseline.py               # embeddings + logistic regression
│   ├── model.py, train.py        # fine-tune MiniLM
│   ├── predict.py                # classify a sentence
│   └── voice.py                  # optional: mic -> Whisper -> model -> TTS
└── submissions/<your-name>/      # your work goes here only
    ├── data/train.jsonl
    ├── runs/
    └── notes.md
```

Members: aaron, evan, raiana, sam, utsav, vivian.

## Quick start
```bash
git clone https://github.com/CLAWS-UMICH/AI-Onboarding-2026.git && cd AI-Onboarding-2026
git checkout -b onboarding/<your-name>
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers sentence-transformers scikit-learn
cd submissions/<your-name>

python ../../code/validate.py data/train.jsonl   # until "0 problems"
python ../../code/baseline.py > baseline.txt
python ../../code/train.py
python ../../code/predict.py "open the nav menu"
```

## Submit
Open **one PR** from `onboarding/<your-name>` when everything is done. It must include `data/train.jsonl`, `runs/metrics.json`, `runs/label2id.json`, `baseline.txt` and a filled-in `notes.md`. Only change files in your own folder. Don't commit `model.pt`; `.gitignore` already skips it.
