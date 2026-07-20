"""
realtime_infer.py

Live webcam sign recognition using retrieval-augmented generation:

  1. Extract a hand-landmark embedding from each webcam frame.
  2. Query the Chroma vector DB (built with build_database.py) for the
     nearest stored embeddings -> get candidate sign labels + distances.
  3. Stabilize predictions with a short rolling buffer + majority vote,
     so a single noisy frame doesn't flicker the output.
  4. Every time a *new* stable sign is confirmed, append it to a running
     "gloss sequence" (e.g. ["HELLO", "MY", "NAME", "ALEX"]).
  5. Periodically (or on keypress), send that gloss sequence to Claude,
     which turns the raw sign sequence into a natural sentence -- this
     is the "generation" half of RAG.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python realtime_infer.py

Controls:
    g  -> ask the LLM to turn the current gloss sequence into a sentence
    c  -> clear the current gloss sequence
    q  -> quit
"""

import os
import time
from collections import deque, Counter

import cv2
import chromadb

from embeding_utils import make_hands_detector, frame_to_embedding
from llm_backends import glosses_to_sentence, get_provider_name, is_configured

DB_PATH = "./chroma_db"
COLLECTION_NAME = "sign_embeddings"

# --- Tuning knobs ---
K_NEIGHBORS = 5          # how many nearest neighbors to look at per frame
DISTANCE_THRESHOLD = 0.9  # if the best match is farther than this, treat as "no sign"
VOTE_WINDOW = 10          # frames to average over before confirming a sign
MIN_VOTES_TO_CONFIRM = 6  # how many of the last VOTE_WINDOW frames must agree
CONFIRM_COOLDOWN = 1.5    # seconds before the same sign can be confirmed again


def classify_embedding(collection, embedding):
    """Query Chroma for nearest neighbors and return the majority label
    among them, or None if the best match is too far away to trust."""
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=K_NEIGHBORS,
    )
    if not results["documents"] or not results["documents"][0]:
        return None

    labels = results["documents"][0]
    distances = results["distances"][0]

    if distances[0] > DISTANCE_THRESHOLD:
        return None  # nearest match still isn't close enough

    # Majority vote among the k neighbors
    most_common_label, _ = Counter(labels).most_common(1)[0]
    return most_common_label


def main():
    client_db = chromadb.PersistentClient(path=DB_PATH)
    collection = client_db.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        print("Your database is empty. Run build_database.py first.")
        return

    provider = get_provider_name()
    llm_ready = is_configured(provider)
    if not llm_ready:
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "NVIDIA_API_KEY"
        print(f"Warning: LLM_PROVIDER is '{provider}' but {key_name} is not set. "
              f"The 'g' sentence-generation key won't work, but live "
              f"classification will still run.")
    else:
        print(f"LLM provider: {provider}")

    detector = make_hands_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    recent_votes = deque(maxlen=VOTE_WINDOW)
    gloss_sequence = []
    last_confirmed_label = None
    last_confirmed_time = 0
    current_sentence = ""

    print("=== Live Sign Recognition ===")
    print("Press 'g' to generate a sentence, 'c' to clear, 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        embedding, _ = frame_to_embedding(frame, detector)
        label = classify_embedding(collection, embedding) if embedding is not None else None
        recent_votes.append(label)

        # Check if a sign has been stably detected for long enough to confirm
        vote_counts = Counter(v for v in recent_votes if v is not None)
        if vote_counts:
            top_label, top_count = vote_counts.most_common(1)[0]
            now = time.time()
            if (top_count >= MIN_VOTES_TO_CONFIRM
                    and (top_label != last_confirmed_label
                         or now - last_confirmed_time > CONFIRM_COOLDOWN)):
                gloss_sequence.append(top_label)
                last_confirmed_label = top_label
                last_confirmed_time = now
                print(f"Confirmed sign: {top_label}  |  Sequence so far: {gloss_sequence}")

        # --- Overlay UI ---
        display_label = label if label else "..."
        cv2.putText(frame, f"Current: {display_label}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Sequence: {' '.join(gloss_sequence[-8:])}", (10, 65),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if current_sentence:
            cv2.putText(frame, current_sentence, (10, 460),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Sign RAG - Live Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            gloss_sequence = []
            current_sentence = ""
            print("Cleared sequence.")
        elif key == ord("g"):
            if not llm_ready:
                print(f"No API key set for provider '{provider}' -- can't call the LLM.")
            else:
                print("Generating sentence...")
                try:
                    current_sentence = glosses_to_sentence(gloss_sequence, provider=provider)
                    print(f"-> {current_sentence}")
                except Exception as e:
                    print(f"LLM call failed: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()