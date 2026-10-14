# Labeling rules

Spec for every row in `train.jsonl` and the test sets. Generators and reviewers follow this file.

**Core rule:** label it if exactly one reading fits (using the defaults below). Otherwise `unhandled`.

## Row format

```json
{"text": "uh whats my primary o2 at", "intents": ["get_oxy_pri_storage"], "source": "llm"}
```

- One intent per row, label copied exactly from `intents.json` (case-sensitive).
- Every generated row has `"source": "llm"`.
- 2 to 20 words. Mostly lowercase, little or no punctuation (it comes from speech-to-text).
- No exact or near duplicates (same sentence with one filler word swapped counts as near duplicate).

## Intent groups

### Oxygen tanks

| Intent | Means | Says |
| --- | --- | --- |
| `get_oxy_pri_storage` | primary O2 tank level % | "primary oxygen", "O2 one", "main O2" |
| `get_oxy_sec_storage` | secondary O2 tank level % | "secondary oxygen", "O2 two", "backup O2" |
| `get_oxy_pri_pressure` | primary O2 tank pressure | "primary O2 pressure", "tank one pressure" |
| `get_oxy_sec_pressure` | secondary O2 tank pressure | "secondary O2 pressure", "backup tank pressure" |

Storage = how much is left (level, percent, remaining, how full). Pressure = the word pressure / psi.

### Suit atmosphere

| Intent | Means | Says |
| --- | --- | --- |
| `get_suit_pressure_oxy` | O2 partial pressure inside the suit | "suit oxygen pressure", "suit O2 partial pressure", "ppO2" |
| `get_suit_pressure_co2` | CO2 partial pressure inside the suit | "suit CO2", "ppCO2", "my CO2" |
| `get_suit_pressure_other` | trace / other gases in the suit | "trace gas", "other gases", "contaminants" |
| `get_suit_pressure_total` | total suit pressure | "suit pressure", "total pressure", "suit psi" |
| `get_helmet_pressure_co2` | CO2 measured in the helmet | "helmet CO2", "CO2 in my helmet" |

### Fans, scrubbers, temperature, coolant

| Intent | Means | Says |
| --- | --- | --- |
| `get_fan_pri_rpm` / `get_fan_sec_rpm` | fan speed | "primary fan", "fan one" / "secondary fan", "fan two", "RPM", "spinning" |
| `get_scrubber_a_co2_storage` / `get_scrubber_b_co2_storage` | CO2 held by scrubber A / B | "scrubber A", "scrubber alpha" / "scrubber B", "scrubber bravo" |
| `get_temperature` | suit internal temperature | "temp", "how hot", "how cold" |
| `get_coolant_storage` | coolant level % | "coolant level", "how much coolant" |
| `get_coolant_gas_pressure` / `get_coolant_liquid_pressure` | coolant loop pressure | "coolant gas pressure" / "coolant liquid pressure", "water loop pressure" |

### Body and mission

| Intent | Means | Says |
| --- | --- | --- |
| `get_heart_rate` | heart rate | "heart rate", "pulse", "BPM" |
| `get_oxy_consumption` | O2 use rate | "how fast am I using oxygen", "O2 consumption", "burn rate" |
| `get_co2_production` | CO2 output rate | "CO2 production", "how much CO2 am I putting out" |
| `get_eva_elapsed_time` | time since EVA start | "how long have I been out", "EVA clock", "elapsed time" |
| `get_primary_battery_level` / `get_secondary_battery_level` | battery % | "battery", "batt", "charge", "power"; "one/main" / "two/backup" |
| `get_warnings` | active alerts | "any warnings", "cautions", "alarms", "what's wrong" |

### HUD menus

| Intent | Means | Says |
| --- | --- | --- |
| `open_menu_vitals` | open vitals panel | "pull up vitals", "show biomed" |
| `open_menu_navigation` | open nav / map panel | "open the map", "nav screen" |
| `open_menu_tasks` | open task checklist | "show tasks", "open checklist", "procedures" |
| `open_menu_uia` | open UIA airlock panel | "UIA panel", "umbilical interface" |
| `close_menu` | close current panel | "close it", "dismiss", "clear the screen", "go back" |

### Navigation and tasks (name intents)

| Intent | Means | Example |
| --- | --- | --- |
| `Get_coordinates` | coordinates of a named place | "where is Shadow Crest", "coords for the lander" |
| `Set_navigation_target` | navigate to a named place | "take me to Crater Rim", "set nav to Station 3" |
| `reroute_navigation` | new route, no new destination | "find another way", "reroute", "send me there instead" |
| `Add_waypoint` | add a named waypoint | "drop a waypoint called Alpha-7" |
| `Delete_waypoint` | delete a named waypoint | "remove the Crater Rim waypoint" |
| `Complete_task` | mark a named task done | "mark soil sample complete", "finished the core drill" |

Vary names: places ("Shadow Crest", "the lander", "Station 3", "Alpha-7"), tasks ("soil sample", "panel inspection").

## Confusable pairs (write contrast rows)

Rows that differ by one word, one per side:

- primary / secondary (oxygen storage, oxygen pressure, fan, battery)
- scrubber A / B; coolant gas / liquid
- tank storage vs tank pressure; suit O2 pressure vs O2 tank pressure
- suit CO2 vs helmet CO2 vs scrubber CO2 vs CO2 production
- O2 consumption vs O2 storage ("am I running low" is storage, "how fast am I using it" is consumption)
- `open_menu_navigation` vs `reroute_navigation` vs `Set_navigation_target`
- `Get_coordinates` vs `Set_navigation_target` vs `Add_waypoint`
- `open_menu_tasks` vs `Complete_task`

## Defaults (when the speaker doesn't say which one)

| Said | Label |
| --- | --- |
| "how's my oxygen", "O2?" | `get_oxy_pri_storage` |
| "what's my CO2", "CO2 level" | `get_suit_pressure_co2` |
| "battery", "charge" | `get_primary_battery_level` |
| "fan speed" | `get_fan_pri_rpm` |
| "suit pressure" | `get_suit_pressure_total` |

No natural default, so **leave out of training** (predict.py returns both sides of the pair):
"scrubber CO2" (A or B), "coolant pressure" (gas or liquid).

## Edge cases

| Case | Label | Example |
| --- | --- | --- |
| Noise / filler / phonetic words around a full command | the real intent | "alpha bravo give me primary oxygen" |
| Clipped radio query with one reading | the intent | "heart rate", "O2 two?" |
| Bare word with many readings | `unhandled` | "pressure" |
| Negated | `unhandled` | "don't open the map" |
| Correction | last instruction | "no wait open vitals" → `open_menu_vitals` |
| Past tense / reporting | `unhandled` | "I checked my O2 earlier" |
| Reporting a task as done | `Complete_task` | "finished the soil sample" |
| Question about the system | `unhandled` | "what does scrubber A do" |
| Two commands in one sentence | **leave out of training** (predict.py splits) | "open vitals and check heart rate" |

## Speech style

Mix all of these:

- **Radio / pilot**: clipped, procedure words. "EV1, request primary O2 status", "say again suit pressure", "verify scrubber bravo", "confirm secondary batt", "fan two RPM, over". NATO alphabet, "niner", callsigns (EV1, EV2, Houston, IV).
- **Casual**: "what's my oxygen at", "uh how hot is it in here", "pull up the map real quick".
- **Stressed**: "need my CO2 now", "warnings, any warnings".
- **Speech-to-text artifacts**: no punctuation, "scrubber ACO2", "oh two", "you I A panel".

No Gen Z slang ("lowkey", "no cap", "fr", "bussin").

## unhandled (150-200 rows)

- Radio chatter: "copy that", "standby", "roger moving to site two", "Houston EV1 comm check"
- Small talk: "nice view", "my visor's fogging a bit", "this rock is huge"
- Uses EVA words but isn't a request: "the scrubber looks dusty", "I like the nav display"
- Negated, past tense, system questions (see edge cases)
- Junk: "uh", "hmm okay", fragments, bare ambiguous words
