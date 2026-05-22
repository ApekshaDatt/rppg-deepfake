"""
rPPG Engine Module - Person B (Signal Processing Specialist)

INPUT CONTRACT (from Person A - Face Detection):
    roi_forehead: np.ndarray  - shape (h, w, 3), dtype uint8, BGR
    roi_cheeks:   np.ndarray  - shape (h, w, 3), dtype uint8, BGR
    frame_count:  int         - current frame index (0-based)
    buffer_size:  int         - BUFFER_SIZE from config.py (default 300)

OUTPUT CONTRACT (to Person C - Threat Analyzer + Person D - UI):
    rppg_signal:    np.ndarray  - shape (N,), dtype float64 processed signal
    estimated_bpm:  float       - dominant heart rate x 60 (0.0 if no data)
    is_calibrating: bool        - True for first CALIBRATION_FRAMES frames
    signal_quality: float [0-1] - SNR-based confidence score
    method_used:    str         - "GREEN", "CHROM", "POS", or "STUB"

Person B: implement signal_extractor.py, preprocessor.py, rppg_algorithms.py
then replace this stub's process_signal() with the real implementation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import BUFFER_SIZE, CALIBRATION_FRAMES


def process_signal(
    roi_forehead: np.ndarray,
    roi_cheeks: np.ndarray,
    frame_count: int,
    buffer_size: int = BUFFER_SIZE,
) -> dict:
    """STUB - returns zeroed signal until Person B implements the real module."""
    is_calibrating = frame_count < CALIBRATION_FRAMES
    return {
        "rppg_signal": np.zeros(buffer_size, dtype=np.float64),
        "estimated_bpm": 0.0,
        "is_calibrating": is_calibrating,
        "signal_quality": 0.0,
        "method_used": "STUB",
    }


__all__ = ["process_signal"]
