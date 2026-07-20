"""
build_database.py

Records reference examples for each sign you want to recognize and stores
their embeddings in a local Chroma vector database.

Usage:
    python build_database.py

You'll be prompted for a sign label (e.g. "HELLO"), then given a 3-second
countdown, then it records ~3 seconds of webcam frames while you hold /
perform the sign. Repeat for as many signs as you want. Type "done" at
the label prompt to stop.

Tip: record each sign 3-5 times (from slightly different angles / hand
positions) so the nearest-neighbor lookup is more robust.
"""

from builtins import print
import time
import uuid

import cv2
import chromadb

from embeding_utils import make_hands_detector, frame_to_embedding

DB_PATH = "./chroma_db"
COLLECTION_NAME = "sign_embeddings"
RECORD_SECONDS = 3


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def record_sign(label, cap, detector, collection):
    print(f"\nGet ready to sign '{label}'...")
    for i in (3, 2, 1):
        print(i)
        time.sleep(1)
    print("Recording...")

    start = time.time()
    count = 0
    while time.time() - start < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        embedding, handedness = frame_to_embedding(frame, detector)
        if embedding is not None:
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[label],
                metadatas=[{"sign": label}],
                ids=[str(uuid.uuid4())],
            )
            count += 1

        cv2.putText(frame, f"Recording {label}...", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Build Database", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Stored {count} frames for '{label}'.")


def main():
    collection = get_collection()
    detector = make_hands_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("=== Sign Database Builder ===")
    print("Type a sign label and press Enter to record it.")
    print("Type 'done' to finish.\n")

    while True:
        label = input("Sign label (or 'done'): ").strip()
        if label.lower() == "done":
            break
        if not label:
            continue
        record_sign(label.upper(), cap, detector, collection)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDatabase saved at {DB_PATH}. Total entries: {collection.count()}")


if __name__ == "__main__":
    main()
