"""
ROI Extractor — Person A (System Architect)

Extracts forehead and cheek sub-regions from a detected face bounding box.
These pixel arrays are the primary input to the rPPG Engine (Person B).

ROI Geometry (per playbook specification):
    Forehead: top 30% of face height, center 60% of face width
    Cheeks:   outer thirds of face width, middle 40% of face height
"""

import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    ROI_PADDING,
    ROI_FOREHEAD_HEIGHT_RATIO,
    ROI_FOREHEAD_WIDTH_RATIO,
    ROI_CHEEK_HEIGHT_RATIO,
)


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(value, hi))


def extract_roi(
    frame: np.ndarray,
    face_bbox: tuple,
    padding: int = ROI_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract forehead and cheek ROI sub-images from a face bounding box.

    Args:
        frame: Original BGR frame (H, W, 3), dtype uint8.
        face_bbox: (x, y, w, h) bounding box of detected face in pixel coordinates.
        padding: Extra pixels to shrink ROI from face edges (avoids hair/jaw).

    Returns:
        (roi_forehead, roi_cheeks)
            roi_forehead — np.ndarray (h, w, 3) BGR forehead region.
            roi_cheeks   — np.ndarray (h, w, 3) BGR cheeks region.
            Both will be empty arrays if bbox is invalid.
    """
    if frame is None or frame.size == 0:
        return create_empty_roi(), create_empty_roi()

    x, y, w, h = face_bbox
    if w <= 0 or h <= 0:
        return create_empty_roi(), create_empty_roi()

    frame_h, frame_w = frame.shape[:2]

    # =========================================================================
    # FOREHEAD ROI
    # Geometry: top 30% of face height, center 60% of face width
    # =========================================================================
    # Width: center 60% → skip (100-60)/2 = 20% from each side
    fw_skip = int(w * (1.0 - ROI_FOREHEAD_WIDTH_RATIO) / 2.0)
    fh_height = int(h * ROI_FOREHEAD_HEIGHT_RATIO)

    fore_x1 = _clamp(x + fw_skip + padding, 0, frame_w - 1)
    fore_y1 = _clamp(y + padding, 0, frame_h - 1)
    fore_x2 = _clamp(x + w - fw_skip - padding, fore_x1 + 1, frame_w)
    fore_y2 = _clamp(y + fh_height - padding, fore_y1 + 1, frame_h)

    roi_forehead = frame[fore_y1:fore_y2, fore_x1:fore_x2]

    # =========================================================================
    # CHEEKS ROI
    # Geometry: outer thirds of width, middle 40% of height
    # "Outer thirds" = left third + right third (skip center third)
    # Middle 40% height = skip top 30%, take next 40%
    # =========================================================================
    cheek_y_start = _clamp(y + int(h * 0.30) + padding, 0, frame_h - 1)
    cheek_y_end = _clamp(y + int(h * 0.30) + int(h * ROI_CHEEK_HEIGHT_RATIO) - padding, cheek_y_start + 1, frame_h)

    # Left cheek (left third of face width)
    left_x1 = _clamp(x + padding, 0, frame_w - 1)
    left_x2 = _clamp(x + int(w * 0.33) - padding, left_x1 + 1, frame_w)

    # Right cheek (right third of face width)
    right_x1 = _clamp(x + int(w * 0.67) + padding, 0, frame_w - 1)
    right_x2 = _clamp(x + w - padding, right_x1 + 1, frame_w)

    left_cheek = frame[cheek_y_start:cheek_y_end, left_x1:left_x2]
    right_cheek = frame[cheek_y_start:cheek_y_end, right_x1:right_x2]

    # Combine cheeks by resizing both to same height and concatenating
    if left_cheek.size == 0 and right_cheek.size == 0:
        roi_cheeks = create_empty_roi()
    elif left_cheek.size == 0:
        roi_cheeks = right_cheek
    elif right_cheek.size == 0:
        roi_cheeks = left_cheek
    else:
        # Resize right cheek to same height as left for horizontal concat
        target_h = left_cheek.shape[0]
        if right_cheek.shape[0] != target_h:
            right_cheek = cv2.resize(right_cheek, (right_cheek.shape[1], target_h))
        roi_cheeks = np.concatenate([left_cheek, right_cheek], axis=1)

    return roi_forehead, roi_cheeks


def create_empty_roi() -> np.ndarray:
    """Create an empty (0, 0, 3) ROI array as a safe default."""
    return np.zeros((0, 0, 3), dtype=np.uint8)


def draw_roi_overlay(
    frame: np.ndarray,
    face_bbox: tuple,
    face_detected: bool,
    padding: int = ROI_PADDING,
) -> np.ndarray:
    """Draw face bounding box and ROI region outlines on the frame.

    For debugging: shows exactly which pixels are being sampled.

    Args:
        frame: BGR frame to draw on (in-place copy).
        face_bbox: (x, y, w, h) face bounding box.
        face_detected: True = draw green, False = draw orange (fallback).
        padding: ROI padding value.

    Returns:
        Frame with drawn overlays.
    """
    overlay = frame.copy()
    x, y, w, h = face_bbox

    if w <= 0 or h <= 0:
        return overlay

    # Face bounding box color
    bbox_color = (0, 255, 0) if face_detected else (0, 165, 255)  # Green or Orange
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bbox_color, 2)

    # Forehead ROI — blue
    fw_skip = int(w * (1.0 - ROI_FOREHEAD_WIDTH_RATIO) / 2.0)
    fh_height = int(h * ROI_FOREHEAD_HEIGHT_RATIO)
    fore_x1 = x + fw_skip + padding
    fore_y1 = y + padding
    fore_x2 = x + w - fw_skip - padding
    fore_y2 = y + fh_height - padding
    cv2.rectangle(overlay, (fore_x1, fore_y1), (fore_x2, fore_y2), (255, 100, 0), 1)
    cv2.putText(overlay, "Forehead", (fore_x1, fore_y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    # Cheeks ROI — cyan
    cheek_y_start = y + int(h * 0.30) + padding
    cheek_y_end = y + int(h * 0.30) + int(h * ROI_CHEEK_HEIGHT_RATIO) - padding
    left_x1 = x + padding
    left_x2 = x + int(w * 0.33) - padding
    right_x1 = x + int(w * 0.67) + padding
    right_x2 = x + w - padding
    cv2.rectangle(overlay, (left_x1, cheek_y_start), (left_x2, cheek_y_end), (255, 255, 0), 1)
    cv2.rectangle(overlay, (right_x1, cheek_y_start), (right_x2, cheek_y_end), (255, 255, 0), 1)

    return overlay
