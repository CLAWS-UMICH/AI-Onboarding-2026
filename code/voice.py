"""Stretch goal: microphone -> Whisper -> intent model -> spoken reply.
pip install faster-whisper sounddevice pyttsx3"""
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel

from predict import predict

RATE, SECONDS, MIN_CONF = 16000, 4, 0.6
# Domain words help Whisper spell things the way your training data does.
HINT = "EVA, UIA, CO2, O2, scrubber, coolant, waypoint, RPM, HUD"

stt = WhisperModel("base.en", device="cpu", compute_type="int8")
tts = pyttsx3.init()

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
    if result["confidence"] < MIN_CONF:
        reply = "Sorry, say that again."
    else:
        reply = result["selected_intent"].replace("_", " ")
    tts.say(reply)
    tts.runAndWait()
