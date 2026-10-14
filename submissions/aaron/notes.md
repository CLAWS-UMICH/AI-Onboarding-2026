# aaron: onboarding notes

## Results
- Data: handwritten / synthetic / mix (how did you generate it?):
- Model (see MODELS.md):
- Baseline accuracy (baseline.txt):
- Fine-tuned accuracy / macro-F1 (runs/metrics.json):

## Unseen sentences (10+)
| Sentence | Predicted | Confidence | Correct? |
|---|---|---|---|
|  |  |  |  |

## What failed and why
-

## Stretch (optional)
- Voice pipeline working? Notes:

## Decisions
- **Env:** uv (`pyproject.toml`, `uv.lock`).
- **Model:** `sentence-transformers/all-mpnet-base-v2`. Trained on short sentence pairs, which matches ~10-word commands. ModernBERT-embed dropped: its gains are long context and speed, irrelevant here.
- **Experiments:** MPNet frozen + logistic regression (baseline), MPNet fine-tune, MPNet + SetFit (contrastive), Laya zero-shot (no fine-tuning).
- **Labeling rules:** `data/RULES.md`. Core rule: label it if exactly one reading fits, else `unhandled`.
  - Unspecified readings use natural defaults ("how's my oxygen" -> primary O2 storage, "what's my CO2" -> suit CO2).
  - No natural default ("scrubber CO2", "coolant pressure") and two-command sentences are left out of training; `predict.py` handles them (returns both sides of the pair / splits on "and").
  - Noise around a full command -> real intent. Negated, past tense, system questions -> `unhandled`. "Finished X" -> `Complete_task`.
- **Name intents:** minimal rule-based slot extraction only (onboarding scope).
- **Data generation (no handwritten data):** separate subagents, each blind to the others:
  - Test-writer: `test_clean`, `test_attack`, `test_final` (150 each). Never sees training data, uses its own names.
  - 4 generators split by intent group; prompts force variety on form, speech style (radio/pilot, casual, stressed, speech-to-text errors), synonyms, length, contrast pairs.
  - Critic reviews labels and realism, then near-duplicates between train and test are removed.
- **Improvement loop:** score on tests -> generate data for misses -> retrain. Max 3 rounds or stop when a round adds <1%. `test_final` is scored once at the end; that's the reported number.

## Progress
- [x] uv setup, branch `onboarding/aaron`
- [x] `data/RULES.md`
- [x] Data generation: 2,115 raw train rows (35 intents x 55 + 190 unhandled) + 450 test rows (150 each; `test_final` came out ~85% clean-style, ~15% attack-style)
- [x] Critic pass: 2,115 -> 1,997 (118 dropped: unnatural "X not Y" phrasing, templated fillers, statements that aren't requests, near-duplicates; 1 relabel). "EV1 checking X" rows dropped (ambiguous: report vs request).
- [x] `dedup.py`: dropped 80 train rows with MPNet cosine >= 0.95 to a same-label test row (different-label matches are contrast pairs, kept). Final `data/train.jsonl`: 1,917 rows, 44-55 per intent, 173 unhandled. `validate.py`: 0 problems.
- [ ] `common.py`, `baseline.py`, `train.py`, `predict.py`
- [ ] Improvement loop (max 3 rounds)
- [ ] Experiments: SetFit, Laya
- [ ] Final results on `test_final`
