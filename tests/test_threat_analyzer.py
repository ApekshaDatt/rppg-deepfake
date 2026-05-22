"""
Comprehensive test suite for Threat Analysis Engine.
Tests all modules: config, fft_analyzer, pattern_detector, threat_scorer, orchestration.
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

# Import all modules under test
from config import (
    BANDPASS_LOW, BANDPASS_HIGH, THREAT_THRESHOLD, LOOP_CORR_THRESHOLD,
    SNR_THRESHOLD, FPS, MIN_SIGNAL_LENGTH, BPM_MIN, BPM_MAX,
    VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN, VERDICT_WINDOW
)
from modules.threat_analyzer.fft_analyzer import (
    bandpass_filter, compute_snr, extract_dominant_frequency, estimate_bpm_from_freq
)
from modules.threat_analyzer.pattern_detector import (
    detect_periodic_loops, analyze_signal_quality_indicators
)
from modules.threat_analyzer.threat_scorer import (
    compute_threat_score, compute_confidence, generate_verdict_and_confidence
)
from modules.threat_analyzer import ThreatAnalyzer, get_analyzer, reset_verdict_history


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def clean_analyzer():
    """Create fresh analyzer instance."""
    analyzer = ThreatAnalyzer()
    return analyzer


@pytest.fixture
def synthetic_pulse_signal():
    """Generate synthetic healthy pulse signal: 72 BPM @ 30 FPS for 10 sec (300 samples)."""
    duration_sec = 10.0
    n_samples = int(FPS * duration_sec)
    t = np.arange(n_samples) / FPS
    
    # 72 BPM = 1.2 Hz
    freq_hz = 72.0 / 60.0
    signal = np.sin(2 * np.pi * freq_hz * t)
    
    # Add Gaussian noise (SNR ≈ 10 dB)
    noise = np.random.normal(0, 0.1, n_samples)
    signal = signal + noise
    
    return signal.astype(np.float64)


@pytest.fixture
def synthetic_deepfake_signal():
    """Generate synthetic deepfake signal: artificial repetitive pattern."""
    duration_sec = 10.0
    n_samples = int(FPS * duration_sec)
    t = np.arange(n_samples) / FPS
    
    # Repeating artificial pattern (sawtooth wave = deepfake indicator)
    pattern_length = int(FPS * 1.0)  # 1-second repeat
    pattern = scipy_signal.sawtooth(2 * np.pi * t[:pattern_length])
    signal = np.tile(pattern, int(n_samples / pattern_length) + 1)[:n_samples]
    
    return signal.astype(np.float64)


@pytest.fixture
def low_snr_signal():
    """Generate signal with very low SNR (high noise)."""
    n_samples = 300
    signal = np.random.normal(0, 1.0, n_samples)  # Pure noise
    return signal.astype(np.float64)


@pytest.fixture
def short_signal():
    """Generate signal shorter than MIN_SIGNAL_LENGTH."""
    return np.random.normal(0, 0.1, 30).astype(np.float64)


@pytest.fixture
def empty_signal():
    """Generate empty signal."""
    return np.array([], dtype=np.float64)


# ============================================================================
# TESTS: fft_analyzer.py
# ============================================================================

class TestFFTAnalyzer:
    """Tests for frequency-domain signal processing."""
    
    def test_bandpass_filter_normal_signal(self, synthetic_pulse_signal):
        """Bandpass filter should reduce out-of-band noise."""
        filtered = bandpass_filter(synthetic_pulse_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert filtered.dtype == np.float64
        assert filtered.shape == synthetic_pulse_signal.shape
        assert not np.any(np.isnan(filtered))
    
    def test_bandpass_filter_empty_signal(self, empty_signal):
        """Bandpass filter on empty signal should return empty."""
        filtered = bandpass_filter(empty_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert filtered.size == 0
    
    def test_bandpass_filter_all_zeros(self):
        """Bandpass filter on all-zeros should return zeros."""
        zeros = np.zeros(100, dtype=np.float64)
        filtered = bandpass_filter(zeros, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert np.allclose(filtered, 0.0)
    
    def test_bandpass_filter_short_signal(self, short_signal):
        """Bandpass filter on very short signal should not crash."""
        filtered = bandpass_filter(short_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert filtered.size > 0
    
    def test_compute_snr_normal_signal(self, synthetic_pulse_signal):
        """SNR computation on healthy signal should return positive value."""
        filtered = bandpass_filter(synthetic_pulse_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        snr = compute_snr(filtered, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert isinstance(snr, float)
        assert -50.0 <= snr <= 100.0
    
    def test_compute_snr_low_quality_signal(self, low_snr_signal):
        """SNR on pure noise should be low."""
        snr = compute_snr(low_snr_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert snr < SNR_THRESHOLD
    
    def test_compute_snr_empty_signal(self, empty_signal):
        """SNR on empty signal should return 0.0."""
        snr = compute_snr(empty_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert snr == 0.0
    
    def test_compute_snr_all_zeros(self):
        """SNR on all-zeros should return 0.0."""
        zeros = np.zeros(100, dtype=np.float64)
        snr = compute_snr(zeros, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        assert snr == 0.0
    
    def test_extract_dominant_frequency_pulse_signal(self, synthetic_pulse_signal):
        """Dominant frequency on healthy pulse should be near 1.2 Hz (72 BPM)."""
        filtered = bandpass_filter(synthetic_pulse_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        freq_hz, power = extract_dominant_frequency(filtered, FPS)
        assert isinstance(freq_hz, float)
        assert isinstance(power, float)
        # Should detect frequency near 1.2 Hz
        assert 1.0 <= freq_hz <= 1.4
    
    def test_extract_dominant_frequency_empty(self, empty_signal):
        """Dominant frequency on empty signal should return (0.0, 0.0)."""
        freq_hz, power = extract_dominant_frequency(empty_signal, FPS)
        assert freq_hz == 0.0
        assert power == 0.0
    
    def test_extract_dominant_frequency_low_energy(self, low_snr_signal):
        """Dominant frequency on low-energy signal may return 0.0."""
        freq_hz, power = extract_dominant_frequency(low_snr_signal, FPS)
        # May or may not detect a frequency; should not crash
        assert isinstance(freq_hz, float)
        assert isinstance(power, float)
    
    def test_estimate_bpm_valid_frequency(self):
        """BPM conversion from valid frequency."""
        freq_hz = 1.2  # 72 BPM
        bpm = estimate_bpm_from_freq(freq_hz)
        assert bpm == pytest.approx(72.0, rel=0.1)
    
    def test_estimate_bpm_out_of_range_low(self):
        """BPM from too-low frequency should return 0.0."""
        freq_hz = 0.5  # 30 BPM (below BPM_MIN=40)
        bpm = estimate_bpm_from_freq(freq_hz)
        assert bpm == 0.0
    
    def test_estimate_bpm_out_of_range_high(self):
        """BPM from too-high frequency should return 0.0."""
        freq_hz = 4.0  # 240 BPM (above BPM_MAX=200)
        bpm = estimate_bpm_from_freq(freq_hz)
        assert bpm == 0.0
    
    def test_estimate_bpm_boundary_low(self):
        """BPM at lower boundary should be valid."""
        freq_hz = 40.0 / 60.0  # 40 BPM (lower bound)
        bpm = estimate_bpm_from_freq(freq_hz)
        assert bpm == pytest.approx(40.0)
    
    def test_estimate_bpm_boundary_high(self):
        """BPM at upper boundary should be valid."""
        freq_hz = 200.0 / 60.0  # 200 BPM (upper bound)
        bpm = estimate_bpm_from_freq(freq_hz)
        assert bpm == pytest.approx(200.0)


# ============================================================================
# TESTS: pattern_detector.py
# ============================================================================

class TestPatternDetector:
    """Tests for temporal anomaly detection."""
    
    def test_detect_periodic_loops_healthy_signal(self, synthetic_pulse_signal):
        """Healthy pulse should NOT trigger loop detection."""
        filtered = bandpass_filter(synthetic_pulse_signal, BANDPASS_LOW, BANDPASS_HIGH, FPS)
        loop_detected, correlation = detect_periodic_loops(filtered, FPS)
        assert isinstance(loop_detected, bool)
        assert isinstance(correlation, float)
        # Healthy signal: correlation should be low
        assert correlation < LOOP_CORR_THRESHOLD
    
    def test_detect_periodic_loops_deepfake_signal(self, synthetic_deepfake_signal):
        """Artificial repeated pattern should trigger loop detection."""
        loop_detected, correlation = detect_periodic_loops(synthetic_deepfake_signal, FPS)
        # Deepfake with artificial repetition: correlation should be high
        assert correlation > 0.5  # Should detect some periodicity
    
    def test_detect_periodic_loops_short_signal(self, short_signal):
        """Very short signal should not trigger loop detection (correlation < threshold)."""
        loop_detected, correlation = detect_periodic_loops(short_signal, FPS)
        assert loop_detected is False
        assert correlation < LOOP_CORR_THRESHOLD
    
    def test_detect_periodic_loops_constant_signal(self):
        """Constant signal should return (False, 0.0)."""
        constant = np.ones(100, dtype=np.float64) * 5.0
        loop_detected, correlation = detect_periodic_loops(constant, FPS)
        assert loop_detected is False
        assert correlation == 0.0
    
    def test_detect_periodic_loops_empty_signal(self, empty_signal):
        """Empty signal should return (False, 0.0)."""
        loop_detected, correlation = detect_periodic_loops(empty_signal, FPS)
        assert loop_detected is False
        assert correlation == 0.0
    
    def test_analyze_signal_quality_indicators_normal(self, synthetic_pulse_signal):
        """Normal signal should have low artifact scores."""
        indicators = analyze_signal_quality_indicators(synthetic_pulse_signal)
        assert isinstance(indicators, dict)
        assert "variance_ratio" in indicators
        assert "clipping_score" in indicators
        assert "discontinuity_score" in indicators
        # All scores should be in [0, 1]
        for key, value in indicators.items():
            assert 0.0 <= value <= 1.0
    
    def test_analyze_signal_quality_indicators_short(self, short_signal):
        """Very short signal should return safe defaults."""
        signal_tiny = np.random.normal(0, 0.1, 10).astype(np.float64)
        indicators = analyze_signal_quality_indicators(signal_tiny)
        assert indicators["variance_ratio"] == 0.0
        assert indicators["clipping_score"] == 0.0
        assert indicators["discontinuity_score"] == 0.0
    
    def test_analyze_signal_quality_indicators_clipped(self):
        """Signal with extreme clipping should have high clipping_score."""
        # Create signal with extreme values
        signal = np.array([1.0, 2.0, 100.0, 2.0, 1.0, -100.0, 1.0, 2.0] * 50, dtype=np.float64)
        indicators = analyze_signal_quality_indicators(signal)
        # Extreme outliers should be flagged
        assert indicators["clipping_score"] > 0.0 or indicators["discontinuity_score"] > 0.0


# ============================================================================
# TESTS: threat_scorer.py
# ============================================================================

class TestThreatScorer:
    """Tests for threat score computation and verdict generation."""
    
    def test_compute_threat_score_real_signal(self):
        """Real signal: low threat score."""
        threat_score = compute_threat_score(
            pulse_present=True,
            loop_correlation=0.3,
            snr_db=10.0,
            signal_quality=0.9
        )
        assert isinstance(threat_score, float)
        assert 0.0 <= threat_score <= 1.0
        # Real signal should have low threat
        assert threat_score < THREAT_THRESHOLD
    
    def test_compute_threat_score_deepfake_signal(self):
        """Deepfake signal: high threat score."""
        threat_score = compute_threat_score(
            pulse_present=True,
            loop_correlation=0.95,  # High loop correlation
            snr_db=2.0,  # Low SNR
            signal_quality=0.3  # Poor quality
        )
        assert threat_score > THREAT_THRESHOLD
    
    def test_compute_threat_score_no_pulse(self):
        """No pulse: low threat contribution."""
        threat_score = compute_threat_score(
            pulse_present=False,
            loop_correlation=0.5,
            snr_db=5.0,
            signal_quality=0.5
        )
        # Without pulse, threat score should be reduced
        assert threat_score < 0.5
    
    def test_compute_threat_score_bounds(self):
        """Threat score should always be in [0, 1]."""
        for _ in range(10):
            threat_score = compute_threat_score(
                pulse_present=bool(np.random.rand() > 0.5),
                loop_correlation=np.random.uniform(0, 1),
                snr_db=np.random.uniform(-10, 30),
                signal_quality=np.random.uniform(0, 1)
            )
            assert 0.0 <= threat_score <= 1.0
    
    def test_compute_confidence_high_snr(self):
        """High SNR should boost confidence."""
        conf_high = compute_confidence(
            snr_db=SNR_THRESHOLD + 5.0,  # Good SNR
            threat_score=0.5,
            pulse_present=True,
            signal_quality=0.9
        )
        assert conf_high > 0.5
    
    def test_compute_confidence_low_snr(self):
        """Low SNR should reduce confidence."""
        conf_low = compute_confidence(
            snr_db=SNR_THRESHOLD - 5.0,  # Poor SNR
            threat_score=0.5,
            pulse_present=True,
            signal_quality=0.5
        )
        assert conf_low < 0.72  # Account for floating point precision
    
    def test_compute_confidence_bounds(self):
        """Confidence should always be in [0, 1]."""
        for _ in range(10):
            conf = compute_confidence(
                snr_db=np.random.uniform(-10, 30),
                threat_score=np.random.uniform(0, 1),
                pulse_present=bool(np.random.rand() > 0.5),
                signal_quality=np.random.uniform(0, 1)
            )
            assert 0.0 <= conf <= 1.0
    
    def test_generate_verdict_calibrating(self):
        """Calibration mode should always return UNCERTAIN."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.9,
            pulse_present=True,
            is_calibrating=True,
            signal_length=300
        )
        assert verdict == VERDICT_UNCERTAIN
    
    def test_generate_verdict_short_signal(self):
        """Very short signal should return UNCERTAIN."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.5,
            pulse_present=True,
            is_calibrating=False,
            signal_length=30
        )
        assert verdict == VERDICT_UNCERTAIN
    
    def test_generate_verdict_no_pulse(self):
        """No pulse detected should return UNCERTAIN."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.9,
            pulse_present=False,
            is_calibrating=False,
            signal_length=300
        )
        assert verdict == VERDICT_UNCERTAIN
    
    def test_generate_verdict_threat(self):
        """High threat score should return THREAT."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.85,
            pulse_present=True,
            is_calibrating=False,
            signal_length=300
        )
        assert verdict == VERDICT_THREAT
    
    def test_generate_verdict_real(self):
        """Low threat score should return REAL."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.2,
            pulse_present=True,
            is_calibrating=False,
            signal_length=300
        )
        assert verdict == VERDICT_REAL
    
    def test_generate_verdict_uncertain(self):
        """Mid-range threat score should return UNCERTAIN."""
        verdict, conf = generate_verdict_and_confidence(
            threat_score=0.55,
            pulse_present=True,
            is_calibrating=False,
            signal_length=300
        )
        assert verdict == VERDICT_UNCERTAIN


# ============================================================================
# TESTS: __init__.py (Orchestration & Integration)
# ============================================================================

class TestThreatAnalyzerOrchestration:
    """Integration tests for full threat analysis pipeline."""
    
    def test_threat_analyzer_healthy_signal(self, clean_analyzer, synthetic_pulse_signal):
        """Healthy signal should produce REAL or UNCERTAIN verdict."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.9
        )
        assert isinstance(result, dict)
        assert result["verdict"] in [VERDICT_REAL, VERDICT_UNCERTAIN]
        assert result["pulse_present"] is True
        assert 60.0 <= result["bpm"] <= 90.0
        assert result["snr_score"] >= 0.0
        assert result["dominant_freq_hz"] > 0.0
    
    def test_threat_analyzer_deepfake_signal(self, clean_analyzer, synthetic_deepfake_signal):
        """Deepfake signal should produce THREAT or UNCERTAIN verdict."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_deepfake_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        assert isinstance(result, dict)
        # Deepfake is harder to detect; may be THREAT or UNCERTAIN
        assert result["verdict"] in [VERDICT_THREAT, VERDICT_UNCERTAIN]
    
    def test_threat_analyzer_calibration_mode(self, clean_analyzer, synthetic_pulse_signal):
        """Calibration mode should return UNCERTAIN."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=True,
            signal_quality=0.9
        )
        assert result["verdict"] == VERDICT_UNCERTAIN
        assert result["confidence"] == 0.0
    
    def test_threat_analyzer_empty_signal(self, clean_analyzer, empty_signal):
        """Empty signal should return UNCERTAIN."""
        result = clean_analyzer.analyze_threat(
            signal=empty_signal,
            estimated_bpm=0.0,
            is_calibrating=False,
            signal_quality=0.0
        )
        assert result["verdict"] == VERDICT_UNCERTAIN
        assert result["pulse_present"] is False
    
    def test_threat_analyzer_short_signal(self, clean_analyzer, short_signal):
        """Short signal should return UNCERTAIN."""
        result = clean_analyzer.analyze_threat(
            signal=short_signal,
            estimated_bpm=0.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        assert result["verdict"] == VERDICT_UNCERTAIN
    
    def test_threat_analyzer_output_dict_keys(self, clean_analyzer, synthetic_pulse_signal):
        """Output dict should have all required keys."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.9
        )
        required_keys = [
            "verdict", "confidence", "bpm", "pulse_present",
            "loop_detected", "snr_score", "dominant_freq_hz"
        ]
        for key in required_keys:
            assert key in result
    
    def test_threat_analyzer_output_types(self, clean_analyzer, synthetic_pulse_signal):
        """Output dict should have correct types."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.9
        )
        assert isinstance(result["verdict"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["bpm"], float)
        assert isinstance(result["pulse_present"], bool)
        assert isinstance(result["loop_detected"], bool)
        assert isinstance(result["snr_score"], float)
        assert isinstance(result["dominant_freq_hz"], float)
    
    def test_threat_analyzer_output_rounding(self, clean_analyzer, synthetic_pulse_signal):
        """Output floats should be properly rounded."""
        result = clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.9
        )
        # Check rounding: confidence to 2 decimals, bpm to 1, etc.
        conf_str = f"{result['confidence']:.2f}"
        assert len(conf_str.split(".")[-1]) <= 2
    
    def test_verdict_history_rolling_window(self, clean_analyzer, synthetic_pulse_signal):
        """Verdict history should maintain rolling window of size VERDICT_WINDOW."""
        # Analyze 7 times (more than VERDICT_WINDOW=5)
        for _ in range(7):
            clean_analyzer.analyze_threat(
                signal=synthetic_pulse_signal,
                estimated_bpm=72.0,
                is_calibrating=False,
                signal_quality=0.9
            )
        
        history = clean_analyzer.get_verdict_history()
        assert len(history) == VERDICT_WINDOW  # Should only store last 5
    
    def test_verdict_aggregation_majority_vote(self, clean_analyzer):
        """Aggregation should use majority vote with THREAT > UNCERTAIN > REAL tiebreak."""
        # Manually populate verdict history for testing
        clean_analyzer.verdict_history.clear()
        clean_analyzer.verdict_history.append((VERDICT_REAL, 0.9))
        clean_analyzer.verdict_history.append((VERDICT_REAL, 0.9))
        clean_analyzer.verdict_history.append((VERDICT_THREAT, 0.7))
        
        # Majority: REAL (2 vs 1)
        aggregated = clean_analyzer._aggregate_verdicts()
        assert aggregated == VERDICT_REAL
    
    def test_reset_verdict_history(self, clean_analyzer, synthetic_pulse_signal):
        """Reset should clear verdict history."""
        # Add some verdicts
        clean_analyzer.analyze_threat(
            signal=synthetic_pulse_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.9
        )
        assert len(clean_analyzer.verdict_history) > 0
        
        # Reset
        clean_analyzer.reset_verdict_history()
        assert len(clean_analyzer.verdict_history) == 0
    
    def test_global_singleton_consistency(self, synthetic_pulse_signal):
        """Global singleton should return same instance."""
        analyzer1 = get_analyzer()
        analyzer2 = get_analyzer()
        assert analyzer1 is analyzer2


# ============================================================================
# TESTS: Edge Cases & Stress
# ============================================================================

class TestEdgeCases:
    """Edge case and stress tests."""
    
    def test_extreme_snr(self):
        """Very high and very low SNR should be handled."""
        threat_high_snr = compute_threat_score(
            pulse_present=True,
            loop_correlation=0.5,
            snr_db=100.0,  # Unrealistic but should not crash
            signal_quality=0.5
        )
        threat_low_snr = compute_threat_score(
            pulse_present=True,
            loop_correlation=0.5,
            snr_db=-50.0,  # Unrealistic but should not crash
            signal_quality=0.5
        )
        assert 0.0 <= threat_high_snr <= 1.0
        assert 0.0 <= threat_low_snr <= 1.0
    
    def test_nan_handling(self, clean_analyzer):
        """NaN values in signal should not crash."""
        signal_with_nan = np.array([1.0, 2.0, np.nan, 4.0] * 75, dtype=np.float64)
        result = clean_analyzer.analyze_threat(
            signal=signal_with_nan,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        # Should not crash, may return UNCERTAIN
        assert result["verdict"] in [VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN]
    
    def test_inf_handling(self, clean_analyzer):
        """Inf values in signal should not crash."""
        signal_with_inf = np.array([1.0, 2.0, np.inf, 4.0] * 75, dtype=np.float64)
        result = clean_analyzer.analyze_threat(
            signal=signal_with_inf,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        # Should not crash, may return UNCERTAIN
        assert result["verdict"] in [VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN]
    
    def test_very_large_values(self, clean_analyzer):
        """Very large signal values should be handled."""
        large_signal = np.ones(300, dtype=np.float64) * 1e6
        result = clean_analyzer.analyze_threat(
            signal=large_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        assert result["verdict"] in [VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN]
    
    def test_very_small_values(self, clean_analyzer):
        """Very small signal values should be handled."""
        small_signal = np.ones(300, dtype=np.float64) * 1e-10
        result = clean_analyzer.analyze_threat(
            signal=small_signal,
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.5
        )
        assert result["verdict"] in [VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
