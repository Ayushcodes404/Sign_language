# Sign Language Classifier via Retrieval-Augmented Generation

Recognizes hand signs live from a webcam using nearest-neighbor retrieval
over MediaPipe hand-landmark embeddings, then uses an LLM to turn the
detected sign sequence into a natural sentence.

## How it works

1. **`embedding_utils.py`** — Uses MediaPipe's **Tasks API** (`HandLandmarker`)
   — the actively maintained hand-tracking API — to extract 21 landmarks
   per hand (x, y, z). Each hand's landmarks are normalized (centered on
   the wrist, scaled to unit size) so the embedding for a given sign looks
   similar no matter where your hand is in frame or how close to the
   camera. Two hands -> 126-number embedding.

   Note: MediaPipe's older `mp.solutions.hands` API is deprecated and
   broken on recent MediaPipe versions, so this project uses the newer
   Tasks API instead, which requires a downloaded model file (next step).

2. **`build_database.py`** — Lets you record short webcam clips for each
   sign you care about. Every frame's embedding is stored in a local
   Chroma vector database along with the sign's label.

3. **`llm_backends.py`** — Pluggable "generation" step. Choose Google
   Gemini or NVIDIA NIM at runtime via the `LLM_PROVIDER` environment
   variable — no code changes needed to switch.

4. **`speech_utils.py`** — Captures a short clip from your microphone
   using `sounddevice` and transcribes it locally using `faster-whisper`
   (no API key, no internet needed after the model's first download).
   This is the "hear the reply" half of the conversation loop.

5. **`realtime_infer.py`** — Runs live:
   - Embeds each incoming webcam frame the same way.
   - Queries Chroma for the nearest stored embeddings (**retrieval**).
   - Uses a short rolling vote to avoid flicker/noise, and confirms a
     sign once it's been stable for several frames.
   - Builds up a list of confirmed signs (a "gloss sequence").
   - On demand (press `g`), sends that sequence to whichever LLM
     provider you've configured, which converts the raw signs into a
     grammatical sentence (**generation**, grounded only in what was
     actually retrieved).
   - On demand (press `r`), listens on your microphone and transcribes
     the other person's spoken reply, displaying it as text on screen —
     completing the two-way conversation.

This mirrors RAG's structure directly: retrieval narrows down *what was
signed* from a knowledge base of examples, and the LLM handles turning
that structured information into fluent language — it's not guessing
the signs itself, just the phrasing.

## Setup

```bash
pip install -r requirements.txt
python download_model.py
```

The second command downloads `hand_landmarker.task` (~a few MB) into the
project folder — the Tasks API loads the hand-tracking model from this
local file rather than bundling it in the pip package.

Choose your LLM provider (only needed for the 'g' sentence-generation step
— live recognition works without it). API keys go in a `.env` file so you
don't have to re-export them every terminal session:

```bash
cp .env.example .env
```

Then open `.env` and fill in the section for whichever provider you're using:

**Google Gemini** — get a key at https://aistudio.google.com/apikey:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

**NVIDIA NIM** — get a key at https://build.nvidia.com:
```
LLM_PROVIDER=nim
NVIDIA_API_KEY=your_key_here
```

`llm_backends.py` loads `.env` automatically at startup. If you'd rather
not use a file, plain `export LLM_PROVIDER=gemini` / `export
GEMINI_API_KEY=...` in your shell works exactly the same way — `.env` is
just a convenience so the values persist across sessions.

If `LLM_PROVIDER` isn't set, it defaults to `gemini`.

**Important:** don't commit `.env` to version control — it contains your
API keys. Add it to `.gitignore` if you're using git.

## Usage

1. Build your reference database:
   ```bash
   python build_database.py
   ```
   Enter a label (e.g. `HELLO`), do a 3-2-1 countdown, then hold/perform
   the sign for ~3 seconds while it records. Repeat for each sign you
   want recognized. Recording each sign a few times from different
   angles improves accuracy.

2. Run live recognition:
   ```bash
   python realtime_infer.py
   ```
   - `g` — generate a sentence from the signs detected so far
   - `r` — listen for a spoken reply and show it as text on screen
   - `c` — clear the current sequence and reply
   - `q` — quit

### Note on the reply / speech-to-text feature

The `r` key uses `sounddevice` (mic recording) and `faster-whisper`
(local transcription) — no API key needed, and it works offline after
the first run. The first time you press `r`, it downloads the whisper
model (~75MB for the default `base.en`) to a local cache, so that one
run needs internet; every run after that is fully offline.

If transcription is too slow or inaccurate on your machine, set the
`WHISPER_MODEL` env var to change model size:
```bash
export WHISPER_MODEL=tiny.en   # faster, less accurate
export WHISPER_MODEL=small.en  # slower, more accurate
```

## Important limitation (and how to extend)

This MVP treats each sign as a **static hand pose** — it looks at one
frame at a time. That works fine for fingerspelling or signs that don't
involve motion, but many real ASL signs are defined by *movement*
(e.g. "hello" is a small wave, "thank you" moves from chin outward).

To handle motion-based signs, the natural upgrade is:
- Buffer a short sliding window of frames (e.g. 15-30 frames / ~1 second).
- Instead of embedding a single frame, embed the whole window — e.g. by
  concatenating/flattening landmark sequences, or (better) training a
  small sequence encoder (1D-CNN or LSTM) that outputs one embedding per
  window.
- Store and query these window-level embeddings instead of per-frame ones.

The database and retrieval logic stay exactly the same — you'd just
swap `frame_to_embedding` for a `window_to_embedding` function.

## Tuning

In `realtime_infer.py`:
- `DISTANCE_THRESHOLD` — how close a match needs to be to count as a
  recognized sign (lower = stricter).
- `MIN_VOTES_TO_CONFIRM` / `VOTE_WINDOW` — how much stability is required
  before a sign is "confirmed" (higher = fewer false positives, but
  slower to react).
- `CONFIRM_COOLDOWN` — minimum time before the same sign can be logged
  again (prevents one held sign from spamming the sequence).
