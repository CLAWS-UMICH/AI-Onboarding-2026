"""Stretch goal: microphone -> Whisper -> intent model -> spoken reply.
pip install faster-whisper sounddevice pyttsx3"""
import subprocess
import sys

import sounddevice as sd
from faster_whisper import WhisperModel

from predict import predict

RATE, SECONDS = 16000, 4
# Domain words help Whisper spell things the way your training data does.
HINT = "EVA, UIA, CO2, O2, scrubber, coolant, waypoint, RPM, HUD"

stt = WhisperModel("base.en", device="cpu", compute_type="int8")


def speak(text):
    if sys.platform == "darwin":
        subprocess.run(["say", text])   # pyttsx3 can hang on repeat calls on macOS
    else:
        import pyttsx3                  # Windows (SAPI5) / Linux (espeak)
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


while True:
    input("Press Enter, then speak for 4 seconds...")
    audio = sd.rec(RATE * SECONDS, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    segments, _ = stt.transcribe(audio.flatten(), language="en", initial_prompt=HINT)
    text = " ".join(s.text for s in segments).strip()
    if not text:
        continue
    result = predict(text)
    print(result)
    if result["selected_intent"] == "unhandled":
        speak("Sorry, say that again.")
    else:
        speak(result["selected_intent"].replace("_", " "))
