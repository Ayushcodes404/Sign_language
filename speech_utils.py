"""
speech_utils.py

Captures a short clip of audio from the microphone and transcribes it to
text. This completes the conversational loop: you sign -> the LLM turns
your gloss sequence into a sentence; the other person speaks a reply ->
this transcribes it back to text on screen.

Uses the SpeechRecognition library with Google's free Web Speech API for
transcription (no API key needed, but does require an internet
connection since the audio is sent to Google's servers for recognition).
"""

import speech_recognition as sr

_recognizer = sr.Recognizer()


def listen_and_transcribe(timeout=5, phrase_time_limit=8):
    """
    Listens on the default microphone for a single phrase and returns
    the transcribed text.

    timeout: seconds to wait for speech to start before giving up.
    phrase_time_limit: max seconds of speech to capture once it starts.

    Returns the transcribed string on success, or a short status message
    (e.g. "(no speech detected)") if nothing usable was captured.
    """
    try:
        with sr.Microphone() as source:
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return "(no speech detected)"
    except OSError as e:
        return f"(microphone error: {e})"

    try:
        return _recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "(could not understand audio)"
    except sr.RequestError as e:
        return f"(speech recognition service error: {e})"