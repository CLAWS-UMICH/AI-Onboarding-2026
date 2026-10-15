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
- [x] `common.py`: shared load, stratified 85/15 split (seeds 42/43/44), scoring, confusion plot. Self-check passes (1,629 train / 288 val).
- [x] `baseline.py` (frozen MPNet + logistic regression): val 0.903 / clean 0.980 / attack 0.700 accuracy (macro-F1 0.915 / 0.983 / 0.722). Attack misses: speech-to-text spellings ("oh two", "vidals"), corrections ("A no wait B"), negation and past tense.
- [x] `train.py` v1 (head lr 1e-3, 10 epochs, smoothing 0.1), 3 seeds: val 0.927 / clean 0.922 / attack 0.787. Underfit: train acc 0.93 and still rising at epoch 10, no early stop; one-word pairs collapse (fan secondary -> primary, scrubber B -> A). Beats baseline on attack (+8.7) and unhandled, loses on clean (-5.8).
- [x] `predict.py` written (threshold, pair "or", split on "and", stop-word slot extraction, CLI loop).
- [x] Laya zero-shot (`convaiinnovations/laya`, 36 intent descriptions as options): val 0.226 / clean 0.213 / attack 0.227. Misses mostly go to `unhandled` (79 of 118 clean misses) or the first option `get_oxy_pri_storage` (28): label/position bias, matching the model card's warning that base checkpoints are near chance zero-shot. Not worth pursuing without fine-tuning.
- Voice stretch: skipped (onboarding scope).
- [x] Tuning screen (8 runs, seed 42, choose by val macro-F1 only, logs in `runs/tune/`): v1 was undertrained. Raising head lr (1e-3 -> 1e-2), encoder lr (2e-5 -> 5e-5) or epochs (10 -> 20) each fix it: val F1 0.910 -> 0.989-0.994, clean 0.920 -> 1.000, attack 0.780 -> 0.82-0.87. Removing label smoothing hurt (0.896). No run overfit (val loss fell every epoch, train-val gap <= 0.011). Top 5 configs tied within one val example (288 rows); tie rule picks h5 (head 1e-2, enc 5e-5, 20 ep max, best ep 4, gap -0.002). Needs multi-seed confirmation.
- [x] Confirm (3 seeds): H5 val F1 0.993 +- 0.003, clean 0.996, attack 0.864 +- 0.019; H7 val F1 0.993 +- 0.001, clean 1.000, attack 0.880 +- 0.007. Tie on val; H7 wins on lower seed spread. Chosen: `--lr-head 1e-2 --lr-encoder 5e-5 --epochs 10`. Val accuracy vs unhandled threshold is flat from 0.2 to 0.5 (within one val example).
- [x] Official run (H7: `--lr-head 1e-2 --lr-encoder 5e-5`, 10 epochs): val 0.992 acc / 0.993 macro-F1, clean 1.000, attack 0.880 (3 seeds). Final model (5 epochs, all data): clean 0.993, attack 0.860. Threshold 0.5 (val flat 0.2-0.5; "say again" safer than a wrong action).
- [x] Improvement round 1 (one round only):
  - Official model's 21 attack misses grouped: 12 speech-to-text spellings/shorthand, 3 corrections (picked the first mention), 5 bare fragments, 1 negation aside.
  - Sealed honest yardstick first: `test_final_attack.jsonl` (150, blind). Before round: raw 0.820, with threshold 0.813 (`runs/eval_before_r1.txt`). Its misses are never read.
  - New data: `augment.py` (speech-to-text substitution table on clean rows, 178 rows) + LLM generator (60 corrections, 40 fragments, 20 negation asides). Critic kept 278/298; it dropped augmented rows where "two"->"too/to", "one"->"won" or "scrubber a"->"eh" erased the word that separates primary/secondary or A/B.
  - Leak caught: I used test_attack sentences as examples in the generator prompt; `dedup.py` (now also checking `test_final_attack`) removed those copies, plus original rows near-identical to the new sealed set. 24 dropped.
  - `data/train.jsonl`: 1,917 -> 2,171 rows (49-65 per intent, 212 unhandled; 164 tagged `"source": "augment"`). `validate.py`: 0 problems.
  - Threshold 0.5 costs accuracy on noisy input (test_attack raw 0.860 vs 0.827 with threshold): revisit after the round.
  - After retrain (same H7 settings, new data; `runs/eval_after_r1.txt`): **test_final_attack 0.820 -> 0.940 raw (0.813 -> 0.933 with threshold)**, the honest gain (+12). test_attack 0.860 -> 0.933 (steered set, inflated). Clean stays 1.000. Val 0.992 -> 0.984, but val is now a different, harder set (326 rows incl. augmented/correction rows), so not comparable. Threshold now costs <1 point.
- [ ] Experiments: SetFit, Laya
- [ ] Final results on `test_final`
