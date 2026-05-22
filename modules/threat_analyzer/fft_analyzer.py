"""
FFT Analyzer: Frequency-domain signal processing for rPPG threat analysis.

Provides two public functions:
  - analyze_fft(signal, fps)     → spectral heartbeat detection (dominant freq, SNR, BPM)
  - apply_bandpass(signal, fps, low, high) → Butterworth bandpass filter

Used by other modules (pattern_detector, threat_scorer) for pre-filtering and
pulse-presence determination.
"""

import sys
import os
import numpy as np
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Path setup so this module can be executed standalone or imported from any
# working directory.  The project root (two levels up from this file) is
# prepended to sys.path so that `config` is always importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import BANDPASS_LOW, BANDPASS_HIGH, SNR_THRESHOLD, FPS


# ===========================================================================
# Public API
# ===========================================================================

def analyze_fft(signal: np.ndarray, fps: float) -> dict:
    """Apply FFT to *signal* and detect the dominant heartbeat frequency.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series rPPG signal (arbitrary units).
    fps : float
        Sampling rate in frames-per-second.

    Returns
    -------
    dict
        dominant_freq_hz : float   – frequency of the strongest spectral peak
                                      inside the cardiac band (Hz).
        snr_score        : float   – ratio of peak power to mean of remaining
                                      in-band power (linear scale).
        pulse_present    : bool    – True when a valid cardiac peak with
                                      sufficient SNR is detected.
        estimated_bpm    : float   – dominant_freq_hz × 60.
    """

    # ── (a) Guard: empty / too-short signal ────────────────────────────────
    safe_default = {
        "dominant_freq_hz": 0.0,
        "snr_score": 0.0,
        "pulse_present": False,
        "estimated_bpm": 0.0,
    }

    if signal is None or len(signal) < 10:
        return safe_default

    # ── (b) Compute one-sided FFT ──────────────────────────────────────────
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2

    # ── (c) Frequency bins ─────────────────────────────────────────────────
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    # ── (d) Bandpass boolean mask ──────────────────────────────────────────
    band_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)

    # ── (e) No bins in cardiac range → no pulse ────────────────────────────
    if not np.any(band_mask):
        return safe_default

    # ── (f) Dominant peak inside the band ──────────────────────────────────
    band_power = power[band_mask]
    band_freqs = freqs[band_mask]

    peak_idx_in_band = np.argmax(band_power)
    peak_power = band_power[peak_idx_in_band]

    # ── (g) Dominant frequency ─────────────────────────────────────────────
    dominant_freq_hz = float(band_freqs[peak_idx_in_band])

    # ── (h) SNR: peak vs. mean of all *other* in-band bins ─────────────────
    if len(band_power) > 1:
        other_power = np.delete(band_power, peak_idx_in_band)
        mean_other = np.mean(other_power)
        if mean_other < 1e-10:
            mean_other = 1e-10          # avoid division by zero
        snr_score = float(peak_power / mean_other)
    else:
        # Only one bin in the band → compare against epsilon
        snr_score = float(peak_power / 1e-10)

    # ── (i) Pulse decision ─────────────────────────────────────────────────
    in_band = BANDPASS_LOW <= dominant_freq_hz <= BANDPASS_HIGH
    pulse_present = bool(in_band and snr_score > SNR_THRESHOLD)

    # ── (j) Return result dict ─────────────────────────────────────────────
    return {
        "dominant_freq_hz": dominant_freq_hz,
        "snr_score": snr_score,
        "pulse_present": pulse_present,
        "estimated_bpm": dominant_freq_hz * 60.0,
    }


def apply_bandpass(signal: np.ndarray, fps: float,
                   low: float, high: float) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter to *signal*.

    Uses second-order-sections (SOS) representation for numerical stability.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series signal.
    fps : float
        Sampling rate in frames-per-second.
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal (float64).  If the signal is too short for the filter
        order it is returned unchanged.
    """

    order = 4
    # Minimum length required by a 4th-order Butterworth SOS filter:
    # each biquad section needs ≥ 3× padlen;  sosfilt itself needs at
    # least (order * 2 + 1) samples to avoid edge artefacts.  We use a
    # conservative guard.
    min_length = 3 * (2 * order + 1)

    # ── (c) Guard: signal too short for filter order ───────────────────────
    if signal is None or len(signal) < min_length:
        if signal is None:
            return np.array([], dtype=np.float64)
        return signal.astype(np.float64)

    # ── (a) Design Butterworth bandpass (SOS output) ───────────────────────
    sos = butter(order, [low, high], btype='band', fs=fps, output='sos')

    # ── (b) Apply filter ───────────────────────────────────────────────────
    filtered = sosfilt(sos, signal)

    # ── (d) Return as float64 ──────────────────────────────────────────────
    return filtered.astype(np.float64)


# ===========================================================================
# Standalone self-test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FFT Analyzer — self-test suite")
    print("=" * 70)

    duration = 10.0        # seconds
    test_fps = 30.0        # frames per second
    n_samples = int(duration * test_fps)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test 1 — Real cardiac signal: 1.2 Hz sine + Gaussian noise
    # ------------------------------------------------------------------
    print("\n[Test 1] Real signal: 1.2 Hz sine + noise (σ=0.05)")
    real_signal = np.sin(2 * np.pi * 1.2 * t) + np.random.default_rng(42).normal(0, 0.05, n_samples)
    result1 = analyze_fft(real_signal, test_fps)

    freq_ok = abs(result1["dominant_freq_hz"] - 1.2) <= 0.15
    snr_ok = result1["snr_score"] > SNR_THRESHOLD
    pulse_ok = result1["pulse_present"] is True

    test1_pass = freq_ok and snr_ok and pulse_ok
    status = "PASS" if test1_pass else "FAIL"
    if test1_pass:
        passed += 1
    else:
        failed += 1

    print(f"  dominant_freq_hz = {result1['dominant_freq_hz']:.4f}  (expected ≈ 1.2 ± 0.15)  {'✓' if freq_ok else '✗'}")
    print(f"  snr_score        = {result1['snr_score']:.4f}  (threshold {SNR_THRESHOLD})  {'✓' if snr_ok else '✗'}")
    print(f"  pulse_present    = {result1['pulse_present']}  (expected True)  {'✓' if pulse_ok else '✗'}")
    print(f"  estimated_bpm    = {result1['estimated_bpm']:.1f}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Test 2 — Flat signal: all zeros
    # ------------------------------------------------------------------
    print("\n[Test 2] Flat signal: np.zeros")
    flat_signal = np.zeros(n_samples)
    result2 = analyze_fft(flat_signal, test_fps)

    test2_pass = result2["pulse_present"] is False
    status = "PASS" if test2_pass else "FAIL"
    if test2_pass:
        passed += 1
    else:
        failed += 1

    print(f"  dominant_freq_hz = {result2['dominant_freq_hz']:.4f}")
    print(f"  snr_score        = {result2['snr_score']:.4f}")
    print(f"  pulse_present    = {result2['pulse_present']}  (expected False)  {'✓' if test2_pass else '✗'}")
    print(f"  estimated_bpm    = {result2['estimated_bpm']:.1f}")
    print(f"  → {status}")

    # ------------------------------------------------------------------
    # Test 3 — Out-of-band signal: 5.0 Hz sine (above BANDPASS_HIGH=3 Hz)
    # ------------------------------------------------------------------
    print("\n[Test 3] Out-of-band signal: 5.0 Hz sine")
    oob_signal = np.sin(2 * np.pi * 5.0 * t)
    result3 = analyze_fft(oob_signal, test_fps)

    test3_pass = result3["pulse_present"] is False
    status = "PASS" if test3_pass else "FAIL"
    if test3_pass:
        passed += 1
    else:
        failed += 1

    print(f"  dominant_freq_hz = {result3['dominant_freq_hz']:.4f}")
    print(f"  snr_score        = {result3['snr_score']:.4f}")
    print(f"  pulse_present    = {result3['pulse_present']}  (expected False)  {'✓' if test3_pass else '✗'}")
    print(f"  estimated_bpm    = {result3['estimated_bpm']:.1f}")
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
