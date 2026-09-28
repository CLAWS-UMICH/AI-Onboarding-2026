# AI-Onboarding-2026
Onboarding repo for the 2026-27 year, AI subteam.

**Task:** write training data for 36 EVA voice intents (35 commands + `unhandled`), fine-tune a small text model (MiniLM by default, 3 options in `MODELS.md`) to classify them, and test it on sentences it has never seen, including a teammate's data. Voice (Whisper + TTS) is optional.

Full guide: [`claws_intent_onboarding.pdf`](claws_intent_onboarding.pdf)

## Layout
```
.
├── claws_intent_onboarding.pdf   # the guide (read this first)
├── requirements.txt
├── code/                         # shared scripts; never edit
│   ├── intents.json              # the 36 intent labels
│   ├── validate.py               # check your data
│   ├── baseline.py               # embeddings + logistic regression
│   ├── model.py, train.py        # fine-tune the encoder (MODEL env var)
│   ├── predict.py                # classify a sentence / peer test
│   └── voice.py                  # optional: mic -> Whisper -> model -> TTS
└── submissions/<your-name>/      # your work goes here only
    ├── MODELS.md                 # the 3 encoders + how to switch
    ├── intents.json              # copy of code/intents.json
    ├── data/train.jsonl
    ├── runs/
    └── notes.md
```

Members and peer-test partner: aaron → evan → gloria → raiana → sam → utsav → vivian → aaron.

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

python ../../code/validate.py data/train.jsonl   # until "0 problems"
python ../../code/baseline.py                    # writes baseline.txt
python ../../code/train.py
python ../../code/predict.py "open the nav menu"
git fetch origin && python ../../code/predict.py --peer <partner>
```

## Submit
Open **one PR** from `onboarding/<your-name>` when everything is done (push earlier so your partner can peer-test). It must include `data/train.jsonl`, `runs/metrics.json`, `runs/label2id.json`, `baseline.txt` and a filled-in `notes.md`. Only change files in your own folder. `.gitignore` keeps `model.pt` out.

To edit the guide, change the `.tex`, run `latexmk -pdf claws_intent_onboarding.tex`, and commit both files.
