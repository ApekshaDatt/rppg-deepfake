"""
Face Detection Module — Person A (System Architect)

Provides face detection and ROI extraction for the rPPG deepfake detection system.

# =============================================================================
# DATA PACKET SCHEMA (Published by Person A — shared with all team members)
# =============================================================================
#
# DataPacket is the dictionary passed from this module to rPPG Engine (Person B)
# and UI (Person D). ALL team members must agree on this schema.
#
# DataPacket = {
#     "frame":         np.ndarray  — shape (H, W, 3), dtype uint8, BGR color space
#                                    Original camera frame for this timestep
#
#     "face_detected": bool        — True if a face was found in this frame
#                                    False if no face (use last known bbox as fallback)
#
#     "face_bbox":     tuple        — (x, y, w, h) in pixels
#                                    Bounding box of the largest detected face
#                                    (0, 0, 0, 0) if face_detected=False
#
#     "roi_forehead":  np.ndarray  — shape (h, w, 3), dtype uint8, BGR
#                                    Cropped forehead sub-region (top 30% of face,
#                                    center 60% width). May be empty if face lost.
#
#     "roi_cheeks":    np.ndarray  — shape (h, w, 3), dtype uint8, BGR
#                                    Cropped cheek sub-region (outer thirds,
#                                    middle 40% height). May be empty if face lost.
#
#     "timestamp":     float       — time.time() value at frame capture
#                                    Used for FPS measurement and CSV logging
# }
#
# =============================================================================
# USAGE EXAMPLE (Person B — rPPG Engine stub)
# =============================================================================
#
#   from modules.face_detection import detect_face, extract_roi, create_data_packet
#
#   face_detected, face_bbox = detect_face(frame)
#   roi_forehead, roi_cheeks = extract_roi(frame, face_bbox)
#   packet = create_data_packet(frame, face_detected, face_bbox,
#                               roi_forehead, roi_cheeks)
#   # Pass packet to rPPG Engine:
#   rppg_output = process_signal(packet["roi_forehead"], packet["roi_cheeks"])
#
# =============================================================================
"""

from modules.face_detection.face_detector import FaceDetector, detect_face
from modules.face_detection.roi_extractor import extract_roi, create_empty_roi
import time
import numpy as np

__all__ = [
    "FaceDetector",
    "detect_face",
    "extract_roi",
    "create_empty_roi",
    "create_data_packet",
]


def create_data_packet(
    frame: np.ndarray,
    face_detected: bool,
    face_bbox: tuple,
    roi_forehead: np.ndarray,
    roi_cheeks: np.ndarray,
    timestamp: float | None = None,
) -> dict:
    """Create a standardized DataPacket dictionary for inter-module communication.

    Args:
        frame: Original BGR frame from cv2.
        face_detected: True if a face was found in this frame.
        face_bbox: (x, y, w, h) bounding box tuple.
        roi_forehead: Forehead sub-region array (h, w, 3).
        roi_cheeks: Cheeks sub-region array (h, w, 3).
        timestamp: Optional capture time (defaults to time.time()).

    Returns:
        DataPacket dict matching the schema above.
    """
    return {
        "frame": frame,
        "face_detected": face_detected,
        "face_bbox": face_bbox,
        "roi_forehead": roi_forehead,
        "roi_cheeks": roi_cheeks,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }
