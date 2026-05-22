"""
Face Detector — Person A (System Architect)

Detects human faces in a video frame using OpenCV's Haar Cascade classifier.
Returns the largest detected face bounding box.
Provides a stateful FaceDetector class with 1-frame tracking fallback.
"""

import cv2
import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import HAAR_SCALE_FACTOR, HAAR_MIN_NEIGHBORS, MIN_FACE_SIZE


def _load_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade XML from OpenCV's built-in data directory."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(
            f"Haar cascade not found at: {cascade_path}\n"
            "Ensure opencv-python is installed correctly."
        )
    return cascade


# Module-level cascade (loaded once at import time)
_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    """Get the cascade classifier, loading it once on first call."""
    global _cascade
    if _cascade is None:
        _cascade = _load_cascade()
    return _cascade


def detect_face(
    frame: np.ndarray,
    scale_factor: float = HAAR_SCALE_FACTOR,
    min_neighbors: int = HAAR_MIN_NEIGHBORS,
    min_face_size: tuple = MIN_FACE_SIZE,
) -> tuple[bool, tuple]:
    """Detect the largest face in a BGR video frame.

    Args:
        frame: np.ndarray of shape (H, W, 3), dtype uint8, BGR color.
        scale_factor: Haar cascade image scale factor (default from config).
        min_neighbors: Minimum neighbors for detection (default from config).
        min_face_size: Minimum face size tuple (w, h) in pixels.

    Returns:
        (face_detected: bool, face_bbox: tuple)
            face_detected — True if at least one face was found.
            face_bbox     — (x, y, w, h) of the largest detected face.
                            (0, 0, 0, 0) if no face detected.
    """
    if frame is None or frame.size == 0:
        return False, (0, 0, 0, 0)

    cascade = _get_cascade()

    # Convert to grayscale for detection
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return False, (0, 0, 0, 0)

    # Equalize histogram to improve detection in varying lighting
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return False, (0, 0, 0, 0)

    # Return the LARGEST face (by area)
    largest = max(faces, key=lambda b: b[2] * b[3])
    x, y, w, h = int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])
    return True, (x, y, w, h)


class FaceDetector:
    """Stateful face detector with 1-frame tracking fallback.

    Maintains the last known bounding box. If face detection fails on a frame,
    the last known bbox is reused to prevent signal interruption in the rPPG pipeline.
    This follows the playbook specification for Person A.
    """

    def __init__(
        self,
        scale_factor: float = HAAR_SCALE_FACTOR,
        min_neighbors: int = HAAR_MIN_NEIGHBORS,
        min_face_size: tuple = MIN_FACE_SIZE,
        max_fallback_frames: int = 5,
    ) -> None:
        """Initialize detector with given parameters.

        Args:
            scale_factor: Haar scale factor (1.05 = slower but more sensitive).
            min_neighbors: Minimum neighbors (3 = more detections, 7 = more strict).
            min_face_size: Minimum (w, h) for a valid face detection.
            max_fallback_frames: How many consecutive frames to allow bbox reuse.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.max_fallback_frames = max_fallback_frames

        self._last_bbox: tuple = (0, 0, 0, 0)
        self._fallback_count: int = 0

    def detect(self, frame: np.ndarray) -> tuple[bool, tuple]:
        """Detect face with 1-frame fallback.

        If detection fails this frame but we have a recent bbox, reuse it.
        Marks face_detected=False even when returning fallback bbox (caller can
        distinguish real detections from fallbacks).

        Args:
            frame: BGR frame from cv2.VideoCapture.

        Returns:
            (face_detected, face_bbox)
                face_detected — True only if detected this frame (not from fallback).
                face_bbox     — Always the best available bbox.
        """
        face_detected, bbox = detect_face(
            frame,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors,
            min_face_size=self.min_face_size,
        )

        if face_detected:
            # Update last known good bbox
            self._last_bbox = bbox
            self._fallback_count = 0
            return True, bbox
        else:
            # Use fallback if within limit
            self._fallback_count += 1
            if self._fallback_count <= self.max_fallback_frames and self._last_bbox != (0, 0, 0, 0):
                # Return last known bbox as fallback (face_detected=False signals fallback)
                return False, self._last_bbox
            else:
                # Exceeded fallback limit or no prior bbox
                return False, (0, 0, 0, 0)

    def reset(self) -> None:
        """Reset tracking state (call on source change or new subject)."""
        self._last_bbox = (0, 0, 0, 0)
        self._fallback_count = 0
