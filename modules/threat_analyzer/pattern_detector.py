"""
Pattern Detector: Temporal anomaly detection for rPPG signals.
Detects artificial periodic loops and signal artifacts indicative of deepfakes.
"""

import numpy as np
from config import LOOP_CORR_THRESHOLD, MIN_AUTOCORR_LENGTH


def detect_periodic_loops(signal: np.ndarray, fs: int) -> tuple[bool, float]:
    """Detect artificial periodic loops via autocorrelation. Returns (loop_detected, max_secondary_correlation)."""
    # Guard: insufficient signal length
    if signal.size < MIN_AUTOCORR_LENGTH:
        return (False, 0.0)
    
    # Guard: constant or all-zero signal
    if np.allclose(signal, 0.0) or np.allclose(signal, signal[0]):
        return (False, 0.0)
    
    try:
        # Normalize signal to zero mean and unit variance for autocorrelation
        signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Compute autocorrelation using FFT (faster than direct method)
        n = len(signal_normalized)
        fft_signal = np.fft.fft(signal_normalized, n=2*n)
        autocorr = np.fft.ifft(fft_signal * np.conj(fft_signal)).real[:n]
        autocorr = autocorr / autocorr[0]  # Normalize by autocorr at lag 0
        
        # Search for secondary peaks beyond lag > fs*0.5 (look for periodicity)
        # This detects if signal repeats itself unnaturally
        min_lag = max(MIN_AUTOCORR_LENGTH, int(fs * 0.5))
        
        if min_lag >= len(autocorr):
            return (False, 0.0)
        
        # Find maximum correlation in the secondary region
        secondary_region = autocorr[min_lag:]
        if secondary_region.size == 0:
            return (False, 0.0)
        
        max_secondary_corr = float(np.max(np.abs(secondary_region)))
        
        # Loop detected if secondary correlation exceeds threshold
        loop_detected = max_secondary_corr > LOOP_CORR_THRESHOLD
        
        return (loop_detected, max_secondary_corr)
    
    except Exception as e:
        print(f"Warning: detect_periodic_loops failed: {e}")
        return (False, 0.0)


def analyze_signal_quality_indicators(signal: np.ndarray) -> dict[str, float]:
    """Detect artifacts: clipping, discontinuities, unstable variance. Returns dict with quality indicators."""
    # Guard: very short signal
    if signal.size < 20:
        return {
            "variance_ratio": 0.0,
            "clipping_score": 0.0,
            "discontinuity_score": 0.0
        }
    
    # Guard: all-zero signal
    if np.allclose(signal, 0.0):
        return {
            "variance_ratio": 0.0,
            "clipping_score": 0.0,
            "discontinuity_score": 0.0
        }
    
    try:
        # ====================================================================
        # 1. VARIANCE RATIO: Detect unstable amplitude across segments
        # ====================================================================
        n_segments = 4  # Divide signal into 4 segments
        segment_length = len(signal) // n_segments
        
        segment_variances = []
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length if i < n_segments - 1 else len(signal)
            segment = signal[start:end]
            if len(segment) > 1:
                segment_variances.append(np.var(segment))
        
        if segment_variances:
            max_var = np.max(segment_variances)
            min_var = np.min(segment_variances)
            # Variance ratio: high value indicates instability
            variance_ratio = (max_var / (min_var + 1e-10)) if min_var > 0 else 1.0
        else:
            variance_ratio = 0.0
        
        # ====================================================================
        # 2. CLIPPING SCORE: Detect saturated/clipped samples
        # ====================================================================
        signal_std = np.std(signal)
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        # Threshold: values beyond 3.5 sigma are likely clipped
        clipping_threshold = 3.5 * signal_std
        clipped_samples = np.sum((np.abs(signal) > clipping_threshold).astype(float))
        clipping_score = clipped_samples / len(signal)  # Fraction of clipped samples
        
        # ====================================================================
        # 3. DISCONTINUITY SCORE: Detect unnatural jumps between samples
        # ====================================================================
        diffs = np.diff(signal)
        max_jump = np.max(np.abs(diffs))
        mean_jump = np.mean(np.abs(diffs))
        
        # Discontinuity: normalized by signal std (detect jumps larger than expected)
        discontinuity_score = (max_jump / (signal_std + 1e-10)) if signal_std > 0 else 0.0
        # Cap discontinuity score to [0, 1] for interpretability
        discontinuity_score = min(discontinuity_score / 10.0, 1.0)  # Normalized
        
        # ====================================================================
        # Return quality indicators dict
        # ====================================================================
        return {
            "variance_ratio": float(np.clip(variance_ratio / 10.0, 0.0, 1.0)),  # Normalize to [0, 1]
            "clipping_score": float(np.clip(clipping_score, 0.0, 1.0)),
            "discontinuity_score": float(np.clip(discontinuity_score, 0.0, 1.0))
        }
    
    except Exception as e:
        print(f"Warning: analyze_signal_quality_indicators failed: {e}")
        return {
            "variance_ratio": 0.0,
            "clipping_score": 0.0,
            "discontinuity_score": 0.0
        }
