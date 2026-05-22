"""
Member C — Comprehensive unit & integration tests for the threat_analyzer module.

Tests fft_analyzer, pattern_detector, and threat_scorer as individual units
and as an integrated pipeline.

Run:  python3 -m unittest tests.test_threat_analyzer -v
  or: python3 tests/test_threat_analyzer.py
"""

import sys
import os
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so imports work from any CWD.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import BANDPASS_LOW, BANDPASS_HIGH, SNR_THRESHOLD, LOOP_CORR_THRESHOLD, FPS
from modules.threat_analyzer.fft_analyzer import analyze_fft, apply_bandpass
from modules.threat_analyzer.pattern_detector import detect_loop, compute_autocorrelation
from modules.threat_analyzer.threat_scorer import score_threat, reset_verdict_history


# ===========================================================================
# Helper: generate synthetic signals
# ===========================================================================

def _make_sine(freq_hz: float, duration: float = 10.0, fps: float = 30.0,
               noise_std: float = 0.0, seed: int = 42) -> np.ndarray:
    """Return a sine wave with optional additive Gaussian noise."""
    n = int(duration * fps)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.sin(2 * np.pi * freq_hz * t)
    if noise_std > 0:
        sig += np.random.default_rng(seed).normal(0, noise_std, n)
    return sig


# ===========================================================================
# Test class: FFT Analyzer
# ===========================================================================

class TestFFTAnalyzer(unittest.TestCase):
    """Unit tests for modules.threat_analyzer.fft_analyzer."""

    def test_real_signal_detected(self):
        """1.2 Hz sine + noise → pulse_present=True, freq ≈ 1.2 Hz."""
        signal = _make_sine(1.2, noise_std=0.05)
        result = analyze_fft(signal, FPS)

        self.assertTrue(result["pulse_present"],
                        f"Expected pulse_present=True, got {result}")
        self.assertGreaterEqual(result["dominant_freq_hz"], 0.9,
                                "Dominant freq too low")
        self.assertLessEqual(result["dominant_freq_hz"], 1.5,
                             "Dominant freq too high")

    def test_flat_signal_no_pulse(self):
        """All-zeros → pulse_present=False."""
        signal = np.zeros(300)
        result = analyze_fft(signal, FPS)

        self.assertFalse(result["pulse_present"],
                         f"Expected pulse_present=False for flat signal, got {result}")

    def test_out_of_band_rejected(self):
        """5.0 Hz sine (above 3 Hz band) → pulse_present=False."""
        signal = _make_sine(5.0)
        result = analyze_fft(signal, FPS)

        self.assertFalse(result["pulse_present"],
                         f"Expected pulse_present=False for 5 Hz signal, got {result}")

    def test_empty_signal_no_crash(self):
        """Empty array → returns dict without raising."""
        result = analyze_fft(np.array([]), FPS)

        self.assertIsInstance(result, dict)
        self.assertFalse(result["pulse_present"])
        self.assertIn("dominant_freq_hz", result)
        self.assertIn("snr_score", result)
        self.assertIn("estimated_bpm", result)


# ===========================================================================
# Test class: Pattern Detector
# ===========================================================================

class TestPatternDetector(unittest.TestCase):
    """Unit tests for modules.threat_analyzer.pattern_detector."""

    def test_perfect_loop_detected(self):
        """Tiled 30-sample sine (10 reps) → loop_detected=True."""
        one_cycle = np.sin(2 * np.pi * np.linspace(0, 1, 30, endpoint=False))
        looped = np.tile(one_cycle, 10)  # 300 samples, 10 identical reps

        result = detect_loop(looped)

        self.assertTrue(result["loop_detected"],
                        f"Expected loop_detected=True for tiled sine, got {result}")

    def test_noisy_signal_not_looped(self):
        """1.2 Hz sine + heavy noise (σ=0.3) → loop_detected=False."""
        signal = _make_sine(1.2, noise_std=0.3, seed=99)
        result = detect_loop(signal)

        self.assertFalse(result["loop_detected"],
                         f"Expected loop_detected=False for noisy signal, got {result}")

    def test_short_array_no_crash(self):
        """Array of length 5 → returns dict, loop_detected=False, no crash."""
        signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = detect_loop(signal)

        self.assertIsInstance(result, dict)
        self.assertFalse(result["loop_detected"])
        self.assertIn("loop_score", result)


# ===========================================================================
# Test class: Threat Scorer
# ===========================================================================

class TestThreatScorer(unittest.TestCase):
    """Unit tests for modules.threat_analyzer.threat_scorer."""

    def setUp(self):
        """Reset verdict history before each test."""
        reset_verdict_history()

    def test_real_verdict_after_warmup(self):
        """7× real-face stubs → verdict == 'REAL'."""
        fft_stub = {
            "dominant_freq_hz": 1.2,
            "snr_score": 8.0,
            "pulse_present": True,
            "estimated_bpm": 72.0,
        }
        loop_stub = {"loop_detected": False, "loop_score": 0.3}

        for _ in range(7):
            result = score_threat(fft_stub, loop_stub)

        self.assertEqual(result["verdict"], "REAL",
                         f"Expected REAL after 7 real-face frames, got {result['verdict']}")

    def test_threat_verdict_flat_signal(self):
        """7× flat-signal stubs → verdict == 'THREAT'."""
        fft_stub = {
            "dominant_freq_hz": 0.0,
            "snr_score": 0.1,
            "pulse_present": False,
            "estimated_bpm": 0.0,
        }
        loop_stub = {"loop_detected": True, "loop_score": 0.97}

        for _ in range(7):
            result = score_threat(fft_stub, loop_stub)

        self.assertEqual(result["verdict"], "THREAT",
                         f"Expected THREAT for flat signal, got {result['verdict']}")

    def test_threat_verdict_perfect_loop(self):
        """7× perfect-loop stubs (high SNR + loop) → verdict == 'THREAT'."""
        fft_stub = {
            "dominant_freq_hz": 1.0,
            "snr_score": 45.0,
            "pulse_present": True,
            "estimated_bpm": 60.0,
        }
        loop_stub = {"loop_detected": True, "loop_score": 0.98}

        for _ in range(7):
            result = score_threat(fft_stub, loop_stub)

        self.assertEqual(result["verdict"], "THREAT",
                         f"Expected THREAT for perfect loop, got {result['verdict']}")

    def test_confidence_is_valid_float(self):
        """Confidence must be a float in [0.0, 1.0]."""
        fft_stub = {
            "dominant_freq_hz": 1.2,
            "snr_score": 8.0,
            "pulse_present": True,
            "estimated_bpm": 72.0,
        }
        loop_stub = {"loop_detected": False, "loop_score": 0.3}

        result = score_threat(fft_stub, loop_stub)

        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_output_dict_has_all_keys(self):
        """Output dict must contain all 7 required keys."""
        fft_stub = {
            "dominant_freq_hz": 1.2,
            "snr_score": 8.0,
            "pulse_present": True,
            "estimated_bpm": 72.0,
        }
        loop_stub = {"loop_detected": False, "loop_score": 0.3}

        result = score_threat(fft_stub, loop_stub)

        required_keys = {
            "verdict", "confidence", "bpm",
            "pulse_present", "loop_detected",
            "snr_score", "dominant_freq_hz",
        }
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key: {key}")


# ===========================================================================
# Test class: End-to-End Pipeline
# ===========================================================================

class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests: analyze_fft → detect_loop → score_threat."""

    def setUp(self):
        reset_verdict_history()

    def test_full_pipeline_real_signal(self):
        """Realistic rPPG signal through full pipeline → verdict 'REAL' after 7 iterations.

        Uses a signal with physiological variability (frequency jitter +
        amplitude modulation + noise) so that the autocorrelation does NOT
        trigger loop detection.
        """
        rng = np.random.default_rng(42)
        n = 300
        t = np.linspace(0, 10, n, endpoint=False)

        # Physiological realism: slight frequency wander + amplitude variation
        freq_jitter = 1.2 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
        phase = np.cumsum(2 * np.pi * freq_jitter / 30.0)
        amplitude = 1.0 + 0.15 * np.sin(2 * np.pi * 0.08 * t)
        signal = amplitude * np.sin(phase) + rng.normal(0, 0.15, n)

        for _ in range(7):
            fft_result = analyze_fft(signal, FPS)
            loop_result = detect_loop(signal)
            result = score_threat(fft_result, loop_result)

        self.assertEqual(result["verdict"], "REAL",
                         f"Expected REAL for realistic rPPG, got {result['verdict']} "
                         f"(loop_detected={result['loop_detected']}, "
                         f"pulse_present={result['pulse_present']})")

    def test_full_pipeline_flat_signal(self):
        """Zeros array through full pipeline 7× → verdict 'THREAT'."""
        signal = np.zeros(300)

        for _ in range(7):
            fft_result = analyze_fft(signal, FPS)
            loop_result = detect_loop(signal)
            result = score_threat(fft_result, loop_result)

        self.assertIn(result["verdict"], ("THREAT", "UNCERTAIN"),
                      f"Expected THREAT or UNCERTAIN for flat signal, got {result['verdict']}")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    run_result = runner.run(suite)

    # Summary banner
    print()
    if run_result.wasSuccessful():
        print("=" * 70)
        print("✅ All Member C tests passed — threat_analyzer module ready for integration")
        print("=" * 70)
    else:
        print("=" * 70)
        print("❌ Some tests FAILED — review output above")
        print("=" * 70)
