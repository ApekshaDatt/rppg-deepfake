"""
Threat Analyzer Module: Orchestrates rPPG deepfake threat detection.

This module receives pre-processed cardiac signals from Person B (rPPG Engine),
analyzes them for deepfake indicators, and returns threat verdicts to Person D (UI) and main.py.

Pipeline:  signal → apply_bandpass → analyze_fft → detect_loop → score_threat → verdict dict
"""

from typing import Any
import numpy as np

from config import (
    FPS, MIN_SIGNAL_LENGTH, BANDPASS_LOW, BANDPASS_HIGH,
    VERDICT_UNCERTAIN
)

# ============================================================================
# INPUT CONTRACT (from Person B: rPPG Engine)
# ============================================================================
# This module receives:
#
#   signal: np.ndarray
#     - Shape: (N,) where N typically = 300 (10 sec @ 30 FPS)
#     - Dtype: float64 (strict requirement for FFT precision)
#     - Content: Pre-filtered rPPG time-series, typically normalized [-1, 1]
#     - Minimum: 60 samples (2 sec @ 30 FPS)
#     - Constraints: May contain NaN/Inf; handled gracefully
#
#   estimated_bpm: float
#     - Range: typically 40-200 BPM
#     - Role: Reference for comparison (extracted frequency takes precedence)
#     - Used for: Validation and fallback only
#
#   is_calibrating: bool
#     - True: First 150 frames (5 sec @ 30 FPS) = warm-up period
#     - False: Normal analysis mode
#     - Behavior: Calibration mode always returns UNCERTAIN (safe)
#
#   signal_quality: float
#     - Range: [0.0, 1.0] where 1.0 = perfect quality
#     - Source: Computed by Person B (rPPG Engine)
#     - Impact: Affects threat score and confidence computation (10% weight)
#     - Low quality: Reduces threat confidence (safe fallback)
#
#   fs: int (default 30)
#     - Sampling rate in frames per second
#     - Must match rPPG Engine output rate
#
# ============================================================================
# OUTPUT CONTRACT (to Person D: UI & main.py)
# ============================================================================
# This module returns: dict[str, Any] with keys:
#
#   "verdict": str
#     - Values: "REAL" (green), "THREAT" (red), "UNCERTAIN" (yellow)
#     - Meaning:
#       * "REAL": Authentic pulse (no alarm)
#       * "THREAT": Deepfake detected (trigger alarm + logging)
#       * "UNCERTAIN": Insufficient data (safe default, no alarm)
#     - Computed: Majority vote on rolling 5-verdict window
#
#   "confidence": float
#     - Range: [0.0, 1.0]
#     - Precision: 2 decimal places (e.g., 0.92)
#     - Meaning: Certainty in verdict (independent of verdict class)
#     - High (>0.8): SNR good, threat score clear, pulse present
#     - Low (<0.4): Contradictory indicators, poor quality
#
#   "bpm": float
#     - Range: [40.0, 200.0] (valid physiological range)
#     - 0.0: No pulse detected (signal quality issue)
#     - Precision: 1 decimal place (e.g., 72.5)
#     - Source: Extracted from dominant frequency
#
#   "pulse_present": bool
#     - True: Pulse detected, dominant_freq > 0.0, BPM valid
#     - False: No pulse (low SNR, noise, filtering issue)
#     - Critical: If False, verdict should be UNCERTAIN (safety check)
#
#   "loop_detected": bool
#     - True: Autocorrelation(lag) > 0.92 (periodic loop found)
#     - False: Normal cardiac rhythm (no artificial repetition)
#     - Meaning: True = deepfake indicator (synthetic signal repeats)
#     - Weight in scoring: 40% (most reliable deepfake signal)
#
#   "snr_score": float
#     - Units: linear ratio (peak power / mean other in-band power)
#     - Threshold: >= 5.0 = good quality
#     - Precision: 2 decimal places (e.g., 8.34)
#     - Computation: Power in [0.7-3.0 Hz] cardiac band
#
#   "dominant_freq_hz": float
#     - Units: Hz (Hertz)
#     - Range: [0.7-3.0 Hz] (42-180 BPM in cardiac band)
#     - 0.0: No energy in cardiac band (no pulse)
#     - Precision: 3 decimal places (e.g., 1.234)
#     - Source: FFT-based peak detection
#
# ============================================================================

# Import submodule functions
from modules.threat_analyzer.fft_analyzer import analyze_fft, apply_bandpass
from modules.threat_analyzer.pattern_detector import detect_loop, compute_autocorrelation
from modules.threat_analyzer.threat_scorer import score_threat
from modules.threat_analyzer.threat_scorer import reset_verdict_history as _scorer_reset

# ============================================================================
# PUBLIC EXPORTS
# ============================================================================
__all__ = [
    "analyze_threat",
    "reset_verdict_history",
    # Re-export submodule functions for convenience
    "analyze_fft",
    "apply_bandpass",
    "detect_loop",
    "compute_autocorrelation",
    "score_threat",
]


# ============================================================================
# Safe default for guard clauses
# ============================================================================
_SAFE_DEFAULT: dict[str, Any] = {
    "verdict": VERDICT_UNCERTAIN,
    "confidence": 0.0,
    "bpm": 0.0,
    "pulse_present": False,
    "loop_detected": False,
    "snr_score": 0.0,
    "dominant_freq_hz": 0.0,
}


# ============================================================================
# Main pipeline
# ============================================================================

def analyze_threat(
    signal: np.ndarray,
    estimated_bpm: float,
    is_calibrating: bool,
    signal_quality: float,
    fs: int = FPS
) -> dict[str, Any]:
    """Run the full threat-analysis pipeline on *signal*.

    Pipeline: signal → bandpass → FFT → loop detect → score → verdict

    Parameters
    ----------
    signal : np.ndarray
        1-D rPPG time-series (float64).
    estimated_bpm : float
        Reference BPM from rPPG Engine (for logging / fallback only).
    is_calibrating : bool
        True during warm-up period → always returns UNCERTAIN.
    signal_quality : float
        Upstream quality metric [0.0, 1.0] from rPPG Engine.
    fs : int
        Frames per second (default from config).

    Returns
    -------
    dict[str, Any]
        Verdict dictionary matching the output contract above.
    """

    # ── Guard 1: Empty or too-short signal ────────────────────────────────
    if signal is None or signal.size == 0 or signal.size < MIN_SIGNAL_LENGTH:
        return dict(_SAFE_DEFAULT)

    # ── Guard 2: Calibration mode ─────────────────────────────────────────
    if is_calibrating:
        return dict(_SAFE_DEFAULT)

    try:
        # ── Step 1: Bandpass filter ───────────────────────────────────────
        filtered = apply_bandpass(signal, fs, BANDPASS_LOW, BANDPASS_HIGH)

        # ── Step 2: FFT analysis ──────────────────────────────────────────
        fft_result = analyze_fft(filtered, fs)

        # ── Step 3: Loop / periodicity detection ──────────────────────────
        loop_result = detect_loop(filtered)

        # ── Step 4: Score + verdict (with rolling majority vote) ──────────
        result = score_threat(fft_result, loop_result)

        return result

    except Exception as e:
        print(f"Error in analyze_threat: {e}")
        return dict(_SAFE_DEFAULT)


def reset_verdict_history() -> None:
    """Clear the rolling verdict window (call on input-source switch)."""
    _scorer_reset()
