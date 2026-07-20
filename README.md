# Sign Language Classifier via Retrieval-Augmented Generation

Recognizes hand signs live from a webcam using nearest-neighbor retrieval
over MediaPipe hand-landmark embeddings, then uses an LLM to turn the
detected sign sequence into a natural sentence.

## How it works

1. **`embedding_utils.py`** — Uses MediaPipe Hands to extract 21 landmarks
   per hand (x, y, z). Each hand's landmarks are normalized (centered on
   the wrist, scaled to unit size) so the embedding for a given sign looks
   similar no matter where your hand is in frame or how close to the
   camera. Two hands -> 126-number embedding.

2. **`build_database.py`** — Lets you record short webcam clips for each
   sign you care about. Every frame's embedding is stored in a local
   Chroma vector database along with the sign's label.

3. **`llm_backends.py`** — Pluggable "generation" step. Choose Google
   Gemini or NVIDIA NIM at runtime via the `LLM_PROVIDER` environment
   variable — no code changes needed to switch.

4. **`realtime_infer.py`** — Runs live:
   - Embeds each incoming webcam frame the same way.
   - Queries Chroma for the nearest stored embeddings (**retrieval**).
   - Uses a short rolling vote to avoid flicker/noise, and confirms a
     sign once it's been stable for several frames.
   - Builds up a list of confirmed signs (a "gloss sequence").
   - On demand (press `g`), sends that sequence to whichever LLM
     provider you've configured, which converts the raw signs into a
     grammatical sentence (**generation**, grounded only in what was
     actually retrieved).

This mirrors RAG's structure directly: retrieval narrows down *what was
signed* from a knowledge base of examples, and the LLM handles turning
that structured information into fluent language — it's not guessing
the signs itself, just the phrasing.

## Setup

```bash
pip install -r requirements.txt
```

Choose your LLM provider (only needed for the 'g' sentence-generation step
— live recognition works without it):

**Google Gemini:**
```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key_here
# optional, defaults to gemini-2.5-flash:
export GEMINI_MODEL=gemini-2.5-flash
```
Get a key at https://aistudio.google.com/apikey

**NVIDIA NIM:**
```bash
export LLM_PROVIDER=nim
export NVIDIA_API_KEY=your_key_here
# optional, defaults to meta/llama-3.1-70b-instruct:
export NIM_MODEL=meta/llama-3.1-70b-instruct
# optional, only needed if self-hosting a NIM microservice instead of
# using NVIDIA's hosted API:
export NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```
Get a key at https://build.nvidia.com

If `LLM_PROVIDER` isn't set, it defaults to `gemini`.

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
   - `c` — clear the current sequence
   - `q` — quit

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
