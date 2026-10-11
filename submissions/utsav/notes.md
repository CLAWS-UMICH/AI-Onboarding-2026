# utsav: onboarding notes

## Results
- Data: synthetic (LLM), 1885 rows, every row reviewed.
- Model: `sentence-transformers/all-mpnet-base-v2`, picked by val score over MiniLM and BGE-small.
- Baseline: 0.929 accuracy, 0.930 macro-F1.
- Fine-tuned: 0.979 accuracy, 0.979 macro-F1 (5-fold CV: 0.985 ± 0.009).
- Overfitting: train 0.999 vs test 0.979; 0.954 on place/task names never seen in training.

Encoder comparison (first data version, 1823 rows):

| Encoder | Baseline acc | Fine-tuned acc |
|---|---|---|
| MiniLM | 0.869 | 0.967 |
| BGE-small | 0.923 | 0.978 |
| MPNet | 0.898 | 0.985 |

## Unseen sentences (handwritten)
| Sentence | Predicted | Confidence | Correct? |
|---|---|---|---|
| something wrong right now? | get_warnings | 0.57 | yes |
| uh, where is Shadow Crest? | Get_coordinates (Shadow Crest) | 0.90 | yes |
| mark this location as Dust Bowl real quick | Add_waypoint (Dust Bowl) | 0.82 | yes |
| send me there instead | unhandled (guess: reroute_navigation) | 0.46 | no |
| finished the core logging | Complete_task (core logging) | 0.86 | yes |
| is it just me, or is it hot in here? whats the temp looking like? | get_temperature | 0.90 | yes |
| close the menu | close_menu | 0.89 | yes |
| uhhhhhhh, im lowkey hungry | unhandled | 0.90 | yes |
| whats my primary battery at? | get_primary_battery_level | 0.86 | yes |
| my hearts beating a little quick, is it fast right now? | get_heart_rate | 0.87 | yes |

9/10 (7/10 before adding data for vague warnings, gibberish and fillers, which these sentences exposed).

## What failed
- Typos: 12% of predictions change after one swapped-letter typo, even with typo rows added.
- "send me there instead": the right guess, but just under the 0.50 threshold.
- A bare name ("Crater Rim") or bare "scrubber"/"coolant" gets a confident answer though it isn't a full command.
- Test misses: "hide my vitals", and "fan two"/"batt one" shorthand mixed up as primary vs secondary.

## Stretch
- Not attempted.
