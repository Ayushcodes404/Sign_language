"""
embedding_utils.py

Turns a webcam frame into a fixed-length numeric "embedding" using
MediaPipe's Tasks API HandLandmarker (the actively maintained
replacement for the old, now-deprecated `mp.solutions.hands` API).

Layout per hand: 21 landmarks x (x, y, z) = 63 numbers.
With up to 2 hands tracked, the full embedding is 126 numbers
(the second hand's slot is zero-filled if only one hand is visible).

Requires a downloaded model file -- see download_model.py.
"""

import os

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

NUM_LANDMARKS = 21
DIMS_PER_HAND = NUM_LANDMARKS * 3  # x, y, z
EMBEDDING_SIZE = DIMS_PER_HAND * 2  # two hands

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def make_hands_detector(model_path=DEFAULT_MODEL_PATH, max_num_hands=2,
                         min_detection_confidence=0.6):
    """Create a MediaPipe Tasks HandLandmarker instance in IMAGE mode
    (suitable for processing one frame at a time, e.g. from a webcam loop)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run download_model.py first."
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _normalize_one_hand(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples in image-relative coords (0-1
    range, z is relative depth).

    Normalization:
      1. Translate so the wrist (landmark 0) is the origin.
      2. Scale so the max distance from the wrist to any landmark is 1.
    This makes the embedding invariant to hand position and distance
    from the camera (roughly invariant to overall scale).
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)
    wrist = pts[0].copy()
    pts -= wrist

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    return pts.flatten()  # shape (63,)


def frame_to_embedding(frame_bgr, detector):
    """
    Run the HandLandmarker on a single BGR frame (as returned by OpenCV)
    and return a fixed-length embedding (numpy array of length
    EMBEDDING_SIZE).

    Returns (embedding, handedness_list) where handedness_list tells you
    which slot ("Left"/"Right") each hand landed in, or (None, None) if
    no hand was detected at all.
    """
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    embedding = np.zeros(EMBEDDING_SIZE, dtype=np.float32)

    if not result.hand_landmarks:
        return None, None

    hands_data = []
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        if result.handedness and i < len(result.handedness):
            label = result.handedness[i][0].category_name  # "Left"/"Right"
        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        hands_data.append((label, coords))

    # Sort so "Left" always lands in slot 0 and "Right" in slot 1 --
    # keeps embeddings consistent across frames.
    hands_data.sort(key=lambda h: h[0])

    handedness_list = []
    for slot, (label, coords) in enumerate(hands_data[:2]):
        normalized = _normalize_one_hand(coords)
        start = slot * DIMS_PER_HAND
        embedding[start:start + DIMS_PER_HAND] = normalized
        handedness_list.append(label)

    return embedding, handedness_list