"""
speech_utils.py

Captures a short clip of audio from the microphone and transcribes it to
text locally using faster-whisper. This completes the conversational
loop: you sign -> the LLM turns your gloss sequence into a sentence;
the other person speaks a reply -> this transcribes it back to text.

Uses `sounddevice` for recording (no PyAudio/PortAudio compilation
headaches) and `faster-whisper` for transcription (runs fully offline
on CPU after the model is downloaded once -- no API key, no internet
needed at inference time, and no dependency on Google's web API).
"""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # whisper expects 16kHz mono audio

# "base.en" is a good balance of speed/accuracy on CPU for English.
# Swap to "small.en" for better accuracy (slower), or "tiny.en" for
# faster/lower accuracy, via the WHISPER_MODEL env var if you want.
import os
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base.en")

_model = None


def _get_model():
    """Lazily load the whisper model (downloads it once on first use,
    caches it locally afterward)."""
    global _model
    if _model is None:
        print(f"Loading whisper model '{MODEL_SIZE}' (first run downloads it)...")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def listen_and_transcribe(duration=5):
    """
    Records `duration` seconds of audio from the default microphone and
    transcribes it. Returns the transcribed string, or a short status
    message (e.g. "(no speech detected)") if nothing usable was heard.
    """
    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        return f"(microphone error: {e})"

    audio = recording.flatten()

    # Skip transcription entirely if the clip is near-silent -- saves
    # time and avoids whisper hallucinating text from noise.
    if np.abs(audio).mean() < 0.001:
        return "(no speech detected)"

    try:
        model = _get_model()
        segments, _info = model.transcribe(audio, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text if text else "(could not understand audio)"
    except Exception as e:
        return f"(transcription error: {e})"
