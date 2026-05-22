"""
Pattern Detector: Temporal anomaly detection for rPPG signals.

Detects whether an rPPG signal is a synthetic mathematical loop — a perfectly
repeating waveform that is the hallmark of a deepfake render engine — by
analysing the normalized autocorrelation for regularly-spaced high-correlation
peaks.

Public API
----------
  detect_loop(signal)              → dict  (loop_detected, loop_score)
  compute_autocorrelation(signal)  → np.ndarray  (full normalized autocorrelation)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows standalone execution and import from any CWD.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import LOOP_CORR_THRESHOLD


# ===========================================================================
# Internal helpers
# ===========================================================================

def _normalized_autocorrelation(normed: np.ndarray) -> np.ndarray:
    """Compute the full normalized autocorrelation of a zero-mean, unit-
    variance signal.

    Each lag is divided by the number of overlapping samples so that the
    coefficients remain in [-1, 1] regardless of lag (unbiased normalisation).
    This is critical for detecting periodicity in finite-length signals where
    a naïve ``correlate`` decays linearly with lag.

    Returns an array of length ``2N - 1`` (negative lags, zero, positive lags).
    """
    n = len(normed)
    full_corr = np.correlate(normed, normed, mode='full')

    # Number of overlapping samples for each lag position
    # For mode='full' the lag index k corresponds to overlap count
    # n - |k - (n-1)|  =  n - |lag|
    overlap_counts = np.array([n - abs(k - (n - 1)) for k in range(2 * n - 1)],
                              dtype=np.float64)
    overlap_counts[overlap_counts == 0] = 1.0  # safety

    full_corr = full_corr / overlap_counts

    # Normalise so that lag-0 == 1.0
    zero_lag = full_corr[n - 1]
    if zero_lag != 0.0:
        full_corr = full_corr / zero_lag

    return full_corr


# ===========================================================================
# Public API
# ===========================================================================

def detect_loop(signal: np.ndarray) -> dict:
    """Compute normalized autocorrelation and detect synthetic periodicity.

    A synthetic (deepfake) signal will exhibit multiple autocorrelation peaks
    above ``LOOP_CORR_THRESHOLD`` at *regular* intervals, indicating a tiled
    or looped waveform.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series rPPG signal.

    Returns
    -------
    dict
        loop_detected : bool  – True when ≥ 2 high-correlation peaks appear
                                 at regularly-spaced lags.
        loop_score    : float – Mean peak coefficient when a loop is detected,
                                 otherwise the max coefficient found.
    """

    # ── (a) Guard: too-short signal ────────────────────────────────────────
    if signal is None or len(signal) < 20:
        return {"loop_detected": False, "loop_score": 0.0}

    # ── (b) Normalise to zero mean, unit variance ─────────────────────────
    std = np.std(signal)
    if std == 0.0:
        return {"loop_detected": False, "loop_score": 0.0}

    normed = (signal - np.mean(signal)) / std

    # ── (c-d) Normalised autocorrelation ──────────────────────────────────
    full_corr = _normalized_autocorrelation(normed)
    n = len(normed)

    # ── (e) Positive lags only (exclude lag 0) ────────────────────────────
    positive_lags = full_corr[n:]  # lags 1, 2, …, N-1

    if len(positive_lags) == 0:
        return {"loop_detected": False, "loop_score": 0.0}

    # ── (f) Find peaks above LOOP_CORR_THRESHOLD ─────────────────────────
    peak_indices = []
    peak_values = []

    for i in range(1, len(positive_lags) - 1):
        val = positive_lags[i]
        if (val > LOOP_CORR_THRESHOLD
                and val >= positive_lags[i - 1]
                and val >= positive_lags[i + 1]):
            peak_indices.append(i)
            peak_values.append(val)

    # ── (g) Decision logic ────────────────────────────────────────────────
    if len(peak_indices) >= 2:
        spacings = np.diff(peak_indices).astype(float)
        mean_spacing = float(np.mean(spacings)) if len(spacings) > 0 else float(peak_indices[0])
        
        # Biometric Frequency Guard:
        # Real heartbeats (spacing < 45 frames) can sometimes be highly stable.
        # To avoid false positives on real humans with clean, resting heart rates,
        # we require an impossibly high mathematical correlation (0.96) and at
        # least 6 stable beats to classify a biological rhythm as a synthetic loop.
        # Macroscopic video loops (>= 45 frames) use the standard config threshold.
        is_biological = mean_spacing < 45.0
        min_required_peaks = 6 if is_biological else 2
        required_corr = 0.96 if is_biological else LOOP_CORR_THRESHOLD
        
        if len(peak_indices) >= min_required_peaks:
            # Re-verify that the peaks actually meet the stricter correlation threshold
            valid_peaks = [val for val in peak_values if val >= required_corr]
            if len(valid_peaks) >= min_required_peaks:
                spacing_variance = np.var(spacings) if len(spacings) > 0 else 0.0
                relative_std = np.sqrt(spacing_variance) / (mean_spacing + 1e-10)
                regular = relative_std < 0.10  # stricter regularity threshold
                if regular:
                    loop_detected = True
                    loop_score = float(np.mean(peak_values))
                else:
                    loop_detected = False
                    loop_score = float(np.max(peak_values))
            else:
                loop_detected = False
                loop_score = float(np.max(peak_values))
        else:
            loop_detected = False
            loop_score = float(np.max(peak_values))
    else:
        # Fallback: not enough peaks → not a loop
        loop_detected = False
        if len(peak_values) > 0:
            loop_score = float(np.max(peak_values))
        elif len(positive_lags) > 0:
            loop_score = float(np.max(positive_lags))
        else:
            loop_score = 0.0

    # ── (h) Return ────────────────────────────────────────────────────────
    return {"loop_detected": loop_detected, "loop_score": loop_score}


def compute_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Return the full normalized autocorrelation array (both lags).

    Useful for plotting by downstream visualisation modules (e.g. Person D's
    plotter).

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series signal.

    Returns
    -------
    np.ndarray
        Normalized autocorrelation with values in [-1, 1].  Length = 2N − 1
        where N = len(signal).  Returns an empty array for degenerate inputs.
    """

    if signal is None or len(signal) < 2:
        return np.array([], dtype=np.float64)

    std = np.std(signal)
    if std == 0.0:
        return np.zeros(2 * len(signal) - 1, dtype=np.float64)

    normed = (signal - np.mean(signal)) / std
    return _normalized_autocorrelation(normed).astype(np.float64)


# ===========================================================================
# Standalone self-test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Pattern Detector — self-test suite")
    print("=" * 70)

    duration = 10.0
    fps = 30.0
    n_samples = int(duration * fps)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.default_rng(42)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test 1 — Perfect loop (deepfake): tiled 1.0 Hz sine
    # ------------------------------------------------------------------
    print("\n[Test 1] Perfect loop: tiled 1.0 Hz sine (deepfake signature)")
    one_cycle = np.sin(2 * np.pi * 1.0 * np.linspace(0, 1, int(fps), endpoint=False))
    looped_signal = np.tile(one_cycle, int(duration))  # 10 full tiles

    result1 = detect_loop(looped_signal)
    test1_pass = (result1["loop_detected"] is True
                  and result1["loop_score"] > LOOP_CORR_THRESHOLD)
    status = "PASS" if test1_pass else "FAIL"
    passed += test1_pass
    failed += (not test1_pass)

    print(f"  loop_detected = {result1['loop_detected']}  (expected True)  "
          f"{'✓' if result1['loop_detected'] else '✗'}")
    print(f"  loop_score    = {result1['loop_score']:.4f}  "
          f"(threshold {LOOP_CORR_THRESHOLD})  "
          f"{'✓' if result1['loop_score'] > LOOP_CORR_THRESHOLD else '✗'}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Test 2 — Real noisy signal: 1.2 Hz sine + heavy noise (σ = 0.2)
    # ------------------------------------------------------------------
    print("\n[Test 2] Real noisy signal: 1.2 Hz sine + noise (σ=0.2)")
    noisy_signal = np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.2, n_samples)

    result2 = detect_loop(noisy_signal)
    test2_pass = result2["loop_detected"] is False
    status = "PASS" if test2_pass else "FAIL"
    passed += test2_pass
    failed += (not test2_pass)

    print(f"  loop_detected = {result2['loop_detected']}  (expected False)  "
          f"{'✓' if not result2['loop_detected'] else '✗'}")
    print(f"  loop_score    = {result2['loop_score']:.4f}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Test 3 — Random noise: np.random.randn(300)
    # ------------------------------------------------------------------
    print("\n[Test 3] Random noise: randn(300)")
    noise_signal = rng.standard_normal(300)

    result3 = detect_loop(noise_signal)
    test3_pass = result3["loop_detected"] is False
    status = "PASS" if test3_pass else "FAIL"
    passed += test3_pass
    failed += (not test3_pass)

    print(f"  loop_detected = {result3['loop_detected']}  (expected False)  "
          f"{'✓' if not result3['loop_detected'] else '✗'}")
    print(f"  loop_score    = {result3['loop_score']:.4f}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Test 4 — Edge case: very short array (length 5)
    # ------------------------------------------------------------------
    print("\n[Test 4] Edge case: array of length 5")
    short_signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

    result4 = detect_loop(short_signal)
    test4_pass = result4["loop_detected"] is False
    status = "PASS" if test4_pass else "FAIL"
    passed += test4_pass
    failed += (not test4_pass)

    print(f"  loop_detected = {result4['loop_detected']}  (expected False)  "
          f"{'✓' if not result4['loop_detected'] else '✗'}")
    print(f"  loop_score    = {result4['loop_score']:.4f}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
