"""
DIRECTIONS -------------------------------------------------------

Stretch goal: microphone -> speech-to-text -> your intent model -> spoken reply.
Write your own version in your folder.

Outline (do it however you like):
- Record a few seconds of audio (e.g. sounddevice).
- Transcribe it (e.g. faster-whisper; a prompt with domain words like
  "EVA, UIA, CO2, scrubber, waypoint" helps spelling).
- Run your predict function on the text.
- Speak a reply (macOS `say`, or pyttsx3 on Windows/Linux).

STEPS -----------------------------------------------------------

1. Push-to-talk: Enter starts recording, Enter again stops (capped at MAX_SECONDS)
2. Whisper (faster-whisper, runs on CPU) turns the audio into text
   - initial_prompt with domain words so it writes "CO2", "UIA", "scrubber" the way our data does
   - vad_filter skips silence, so an empty recording gives no text instead of un made-up sentence hehe
3. predict() from predict.py picks the intent (+ slot for the five name intents)
4. Speak a short reply from REPLIES (no telemetry here, so it says what it would show, not the value)
5. Usage (pip install faster-whisper sounddevice, plus pyttsx3 on Windows/Linux):
     python code/voice.py                  - talk into the mic, Ctrl+C quits
     python code/voice.py clip.wav ...     - transcribe 16-bit .wav files instead (no mic needed)
"""

import subprocess
import sys
import threading

import numpy as np
from faster_whisper import WhisperModel

from predict import predict

RATE = 16000 # Whisper expects 16 kHz mono
MAX_SECONDS = 10
WHISPER_SIZE = "base.en" # "small.en" is more accurate but slower apparently
HINT = "EVA, UIA, HUD, O2, CO2, scrubber A, scrubber B, coolant, waypoint, RPM, airlock, primary, secondary"

# What to say back for each intent; {slot} is filled in for the five name intents
REPLIES = {
    "get_oxy_pri_storage": "Primary oxygen storage.",
    "get_oxy_sec_storage": "Secondary oxygen storage.",
    "get_oxy_pri_pressure": "Primary oxygen tank pressure.",
    "get_oxy_sec_pressure": "Secondary oxygen tank pressure.",
    "get_suit_pressure_oxy": "Suit oxygen pressure.",
    "get_suit_pressure_co2": "Suit C O 2 pressure.",
    "get_suit_pressure_other": "Suit trace gas pressure.",
    "get_suit_pressure_total": "Total suit pressure.",
    "get_helmet_pressure_co2": "Helmet C O 2 pressure.",
    "get_fan_pri_rpm": "Primary fan speed.",
    "get_fan_sec_rpm": "Secondary fan speed.",
    "get_scrubber_a_co2_storage": "Scrubber A storage.",
    "get_scrubber_b_co2_storage": "Scrubber B storage.",
    "get_temperature": "Suit temperature.",
    "get_coolant_storage": "Coolant storage.",
    "get_coolant_gas_pressure": "Coolant gas pressure.",
    "get_coolant_liquid_pressure": "Coolant liquid pressure.",
    "get_heart_rate": "Heart rate.",
    "get_oxy_consumption": "Oxygen consumption rate.",
    "get_co2_production": "C O 2 production rate.",
    "get_eva_elapsed_time": "Elapsed EVA time.",
    "get_primary_battery_level": "Primary battery level.",
    "get_secondary_battery_level": "Secondary battery level.",
    "get_warnings": "Showing active warnings.",
    "open_menu_vitals": "Opening vitals.",
    "open_menu_navigation": "Opening navigation.",
    "open_menu_tasks": "Opening tasks.",
    "open_menu_uia": "Opening U I A panel.",
    "close_menu": "Closing panel.",
    "Get_coordinates": "Coordinates for {slot}.",
    "Set_navigation_target": "Navigating to {slot}.",
    "reroute_navigation": "Calculating a new route.",
    "Add_waypoint": "Adding waypoint {slot}.",
    "Delete_waypoint": "Deleting waypoint {slot}.",
    "Complete_task": "Marking {slot} complete.",
    "unhandled": "Sorry, say that again.",
}


# Reply sentence for a predict()
def reply_for(result):
    template = REPLIES.get(result["selected_intent"], "Sorry, say that again.")
    if "{slot}" in template and not result.get("slot"):
        return "Which one? I didn't get the name"
    return template.format(slot=result.get("slot"))


# Say text out loud: built-in `say` on macOS, pyttsx3 elsewhere
def speak(text):
    if sys.platform == "darwin":
        subprocess.run(["say", text]) # pyttsx3 can hang on repeat calls on macOS
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


# Record from the default mic until Enter is pressed (or MAX_SECONDS hit), return float32 samples
def record():
    import sounddevice as sd # imported here so file mode works without importing mic library

    chunks = []
    stop = threading.Event()

    def on_audio(data, frames, time, status):
        chunks.append(data.copy())
        if sum(len(c) for c in chunks) >= RATE * MAX_SECONDS:
            stop.set()

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=on_audio):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks).flatten() if chunks else np.zeros(0, dtype=np.float32)


# Read a .wav into float32 mono samples at RATE (stdlib only, avoids faster-whisper's own decoder)
def load_wav(path):
    import wave

    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit(f"{path}: only 16-bit WAV is supported")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    audio = audio.reshape(-1, channels).mean(axis=1) # stereo -> mono
    if rate != RATE: # simple linear resample to 16 kHz
        audio = np.interp(np.arange(0, len(audio), rate / RATE), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


# Speech -> text. audio is a float32 array at RATE hehe
def transcribe(stt, audio):
    segments, _ = stt.transcribe(audio, language="en", initial_prompt=HINT, vad_filter=True)
    return " ".join(s.text for s in segments).strip()


# Transcribe, classify, print and speak un utterance
def handle(stt, audio):
    text = transcribe(stt, audio)
    if not text:
        print("(heard nothing)")
        return
    result = predict(text)
    reply = reply_for(result)
    print(result)
    print(f"-> {reply}")
    speak(reply)


def main():
    print(f"loading Whisper {WHISPER_SIZE}...")
    stt = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    if len(sys.argv) > 1: # file mode
        for path in sys.argv[1:]:
            print(f"\n{path}")
            handle(stt, load_wav(path))
        return

    try:
        while True:
            input("\nPress Enter and speak, Enter again to stop (Ctrl+C quits)...")
            print("listening...")
            handle(stt, record())
    except (KeyboardInterrupt, EOFError):
        print("\nbye-bye")


if __name__ == "__main__":
    main()
