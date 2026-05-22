"""
Threat Analyzer Module: Orchestrates rPPG deepfake threat detection.

This module receives pre-processed cardiac signals from Person B (rPPG Engine),
analyzes them for deepfake indicators, and returns threat verdicts to Person D (UI) and main.py.
It maintains stateful verdict history with majority-vote aggregation for temporal smoothing.
"""

from collections import deque
from typing import Any
import numpy as np

from config import (
    VERDICT_WINDOW, FPS, MIN_SIGNAL_LENGTH, BANDPASS_LOW, BANDPASS_HIGH,
    VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN,
    CONFIDENCE_DECIMALS, BPM_DECIMALS, SNR_DECIMALS, FREQ_DECIMALS
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
#     - Units: dB (decibels)
#     - Range: [-50, 100] dB (clamped)
#     - Threshold: >= 5.0 dB = good quality
#     - Precision: 2 decimal places (e.g., 8.34)
#     - Computation: Power in [0.7-3.0 Hz] vs. out-of-band noise
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
from modules.threat_analyzer.fft_analyzer import (
    bandpass_filter, compute_snr, extract_dominant_frequency, estimate_bpm_from_freq
)
from modules.threat_analyzer.pattern_detector import (
    detect_periodic_loops, analyze_signal_quality_indicators
)
from modules.threat_analyzer.threat_scorer import (
    compute_threat_score, compute_confidence, generate_verdict_and_confidence
)

# ============================================================================
# PUBLIC EXPORTS
# ============================================================================
__all__ = [
    "ThreatAnalyzer",
    "analyze_threat",
    "reset_verdict_history",
    "get_analyzer",
]


class ThreatAnalyzer:
    """Stateful threat analyzer with rolling verdict aggregation."""
    
    def __init__(self) -> None:
        """Initialize threat analyzer with empty verdict history."""
        self.verdict_history: deque = deque(maxlen=VERDICT_WINDOW)
    
    def reset_verdict_history(self) -> None:
        """Clear verdict history on input source change."""
        self.verdict_history.clear()
    
    def get_verdict_history(self) -> list[tuple[str, float]]:
        """Return verdict history for inspection/debugging."""
        return list(self.verdict_history)
    
    def _aggregate_verdicts(self) -> str:
        """Aggregate verdicts using majority vote. Tiebreak: THREAT > UNCERTAIN > REAL."""
        if not self.verdict_history:
            return VERDICT_UNCERTAIN  # Safe default
        
        # Count verdict occurrences
        verdict_counts = {VERDICT_REAL: 0, VERDICT_THREAT: 0, VERDICT_UNCERTAIN: 0}
        for verdict_str, _ in self.verdict_history:
            if verdict_str in verdict_counts:
                verdict_counts[verdict_str] += 1
        
        # Majority vote with tiebreak: THREAT > UNCERTAIN > REAL
        max_count = max(verdict_counts.values())
        
        # Tiebreak precedence
        if verdict_counts[VERDICT_THREAT] == max_count:
            return VERDICT_THREAT
        elif verdict_counts[VERDICT_UNCERTAIN] == max_count:
            return VERDICT_UNCERTAIN
        else:
            return VERDICT_REAL
    
    def _safe_default_dict(self, verdict: str, pulse_present: bool) -> dict[str, Any]:
        """Return safe defaults when analysis is skipped or fails."""
        return {
            "verdict": verdict,
            "confidence": 0.0,
            "bpm": 0.0,
            "pulse_present": pulse_present,
            "loop_detected": False,
            "snr_score": 0.0,
            "dominant_freq_hz": 0.0
        }
    
    def analyze_threat(
        self,
        signal: np.ndarray,
        estimated_bpm: float,
        is_calibrating: bool,
        signal_quality: float,
        fs: int = FPS
    ) -> dict[str, Any]:
        """
        Main threat analysis pipeline. Returns verdict dict with confidence, BPM, and spectral metrics.
        
        Inputs:
            signal: np.ndarray shape (N,), dtype float64 — time-series rPPG signal
            estimated_bpm: float — reference BPM from rPPG Engine (for reference)
            is_calibrating: bool — calibration mode flag
            signal_quality: float [0, 1] — signal quality from rPPG Engine
            fs: int — frames per second (default 30)
        
        Returns:
            dict with keys: verdict, confidence, bpm, pulse_present, loop_detected, snr_score, dominant_freq_hz
        """
        # ====================================================================
        # GUARD 1: Empty or too-short signal
        # ====================================================================
        if signal.size == 0:
            return self._safe_default_dict(VERDICT_UNCERTAIN, pulse_present=False)
        
        if signal.size < MIN_SIGNAL_LENGTH:
            return self._safe_default_dict(VERDICT_UNCERTAIN, pulse_present=False)
        
        # ====================================================================
        # GUARD 2: Calibration mode
        # ====================================================================
        if is_calibrating:
            return self._safe_default_dict(VERDICT_UNCERTAIN, pulse_present=False)
        
        try:
            # ================================================================
            # STEP 1: Apply bandpass filter to extract cardiac frequency band
            # ================================================================
            filtered_signal = bandpass_filter(signal, BANDPASS_LOW, BANDPASS_HIGH, fs)
            
            # ================================================================
            # STEP 2: Extract dominant frequency via FFT
            # ================================================================
            dominant_freq_hz, _ = extract_dominant_frequency(filtered_signal, fs)
            
            # ================================================================
            # STEP 3: Convert frequency to BPM and validate
            # ================================================================
            bpm_analyzed = estimate_bpm_from_freq(dominant_freq_hz)
            
            # ================================================================
            # STEP 4: Compute signal-to-noise ratio
            # ================================================================
            snr_db = compute_snr(filtered_signal, BANDPASS_LOW, BANDPASS_HIGH, fs)
            
            # ================================================================
            # STEP 5: Detect periodic loops (deepfake indicator)
            # ================================================================
            loop_detected, loop_correlation = detect_periodic_loops(filtered_signal, fs)
            
            # ================================================================
            # STEP 6: Determine pulse presence
            # ================================================================
            pulse_present = (dominant_freq_hz > 0.0) and (bpm_analyzed > 0.0)
            
            # ================================================================
            # STEP 7: Compute threat score (4-component fusion)
            # ================================================================
            threat_score = compute_threat_score(
                pulse_present=pulse_present,
                loop_correlation=loop_correlation,
                snr_db=snr_db,
                signal_quality=signal_quality
            )
            
            # ================================================================
            # STEP 8: Generate verdict and confidence
            # ================================================================
            verdict_str, confidence = generate_verdict_and_confidence(
                threat_score=threat_score,
                pulse_present=pulse_present,
                is_calibrating=False,
                signal_length=signal.size
            )
            
            # ================================================================
            # STEP 9: Recompute confidence with full information
            # ================================================================
            confidence = compute_confidence(
                snr_db=snr_db,
                threat_score=threat_score,
                pulse_present=pulse_present,
                signal_quality=signal_quality
            )
            
            # ================================================================
            # STEP 10: Add to verdict history and aggregate
            # ================================================================
            self.verdict_history.append((verdict_str, confidence))
            aggregated_verdict = self._aggregate_verdicts()
            
            # ================================================================
            # STEP 11: Round and format output
            # ================================================================
            output_dict = {
                "verdict": aggregated_verdict,
                "confidence": round(confidence, CONFIDENCE_DECIMALS),
                "bpm": round(bpm_analyzed, BPM_DECIMALS),
                "pulse_present": pulse_present,
                "loop_detected": loop_detected,
                "snr_score": round(snr_db, SNR_DECIMALS),
                "dominant_freq_hz": round(dominant_freq_hz, FREQ_DECIMALS)
            }
            
            return output_dict
        
        except Exception as e:
            # Fail safely: return UNCERTAIN verdict on any unexpected error
            print(f"Error in analyze_threat: {e}")
            return self._safe_default_dict(VERDICT_UNCERTAIN, pulse_present=False)


# ============================================================================
# GLOBAL SINGLETON INSTANCE
# ============================================================================
_analyzer_instance: ThreatAnalyzer | None = None


def get_analyzer() -> ThreatAnalyzer:
    """Get or create global ThreatAnalyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ThreatAnalyzer()
    return _analyzer_instance


def analyze_threat(
    signal: np.ndarray,
    estimated_bpm: float,
    is_calibrating: bool,
    signal_quality: float,
    fs: int = FPS
) -> dict[str, Any]:
    """Convenience function: analyze threat using global singleton."""
    analyzer = get_analyzer()
    return analyzer.analyze_threat(signal, estimated_bpm, is_calibrating, signal_quality, fs)


def reset_verdict_history() -> None:
    """Convenience function: reset verdict history on input source change."""
    analyzer = get_analyzer()
    analyzer.reset_verdict_history()
