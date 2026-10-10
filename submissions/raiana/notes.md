# aaron: onboarding notes

## Results
- Data: handwritten / synthetic / mix (how did you generate it?): synthetic
- Model (see MODELS.md): MiniLM
- Baseline accuracy (baseline.txt): 95.3%
- Fine-tuned accuracy / macro-F1 (runs/metrics.json): 97.0%

## Unseen sentences (10+)
| Sentence | Predicted | Confidence | Correct? |
|---|---|---|---|
| how much oxygen do I have | get_oxy_pri_storage | 0.943 | Yes |
| navigate to Olympus Ridge | Set_navigation_target (slot: Olympus Ridge) | 0.856 | Yes |
| drop a pin called rock cluster | Add_waypoint (slot: rock cluster) | 0.911 | Yes |
| get rid of the crater waypoint | Delete_waypoint (slot: crater) | 0.930 | Yes |
| where is the lander | Get_coordinates (slot: lander) | 0.828 | Yes |
| I finished the sample collection | Complete_task (slot: sample collection) | 0.863 | Yes |
| bring up the map | open_menu_navigation | 0.952 | Yes |
| find a new path around the crater | reroute_navigation | 0.872 | Yes |
| how hot is it | get_temperature | 0.948 | Yes |
| how's my heartbeat | get_heart_rate | 0.954 | Yes |
| any alarms | get_warnings | 0.939 | Yes |
| how long have I been out here | get_eva_elapsed_time | 0.955 | Yes |
| what's the second fan spinning at | get_fan_sec_rpm | 0.888 | Yes |
| is scrubber B full | get_scrubber_b_co2_storage | 0.943 | Yes |
| close it | close_menu | 0.945 | Yes |
| pressure | unhandled | 0.214 | Yes (ambiguous, so good it refused) |

## What failed and why
Major failures I saw with the examples I saw them with:

- Slot extraction very difficult, sometimes picks up entire phrase instead of just the one word it's supposed to
    - Get rid of the crater waypoint
- "Am I running low on oxygen?" gave get_oxy_consumption (0.91) but it should return oxy_pri_storage
- One word inputs often giving unhandled

Plan to fix:
- Added "rid" to LEAD strip list in predict.py so "get rid of crater waypoint" now gives slot "crater"
- Added 8 more oxygen questions that don't name a tank, labelled as primary tank storage getting, retrained
- Added 18 one-word / two-word inputs

## Stretch (optional)
Voice pipeline working?
- Yes, used sounddevice for push a button to talk mic
- faster-whisper (with domain-word prompt + VAD filter)
- Then it goes through my fine-tuned model and give sspoken reply
- Also does support taking .wav files as input instead
- tested on four sentences that I said:
  | Said | Whisper heard | Intent | Conf |
  |---|---|---|---|
  | "what is the scrubber A CO2 storage" | "What is the scrubber ACO2 storage?" | get_scrubber_a_co2_storage | 0.67 |
  | "add a waypoint named Olympus Ridge" | "at a waypoint named Olympus Ridge." | Add_waypoint (slot: Olympus Ridge) | 0.81 |
  | "open the UIA panel" | "Open the UIA panel" | open_menu_uia | 0.97 |
  | "what's my heart rate" | "What's my heart rate?" | get_heart_rate | 0.97 |
- Notes: Whisper did merge "scrubber A CO2" into "scrubber ACO2" which made conf drop to 0.67
- Model still got both right, but should add a few more messier sentences 