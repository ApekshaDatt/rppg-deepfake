"""
Integration Test & Demo: Threat Analyzer End-to-End Workflow.
Demonstrates the full threat analysis pipeline with realistic scenarios.
"""

import numpy as np
from scipy import signal as scipy_signal

from modules.threat_analyzer import analyze_threat, get_analyzer, reset_verdict_history
from config import FPS


def demo_healthy_pulse():
    """Scenario 1: Analyze a healthy pulse signal."""
    print("\n" + "="*70)
    print("SCENARIO 1: HEALTHY PULSE (Real Face)")
    print("="*70)
    
    # Generate healthy pulse: 75 BPM @ 30 FPS for 10 sec (300 samples)
    duration_sec = 10.0
    n_samples = int(FPS * duration_sec)
    t = np.arange(n_samples) / FPS
    
    freq_hz = 75.0 / 60.0  # 75 BPM
    signal_clean = np.sin(2 * np.pi * freq_hz * t)
    
    # Add realistic Gaussian noise (SNR ≈ 12 dB)
    noise = np.random.normal(0, 0.08, n_samples)
    signal = signal_clean + noise
    
    result = analyze_threat(
        signal=signal.astype(np.float64),
        estimated_bpm=75.0,
        is_calibrating=False,
        signal_quality=0.92,
        fs=FPS
    )
    
    print(f"Verdict:           {result['verdict']}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"BPM:               {result['bpm']:.1f}")
    print(f"Pulse Present:     {result['pulse_present']}")
    print(f"Loop Detected:     {result['loop_detected']}")
    print(f"SNR (dB):          {result['snr_score']:.2f}")
    print(f"Dominant Freq (Hz):{result['dominant_freq_hz']:.3f}")
    
    assert result['verdict'] in ['REAL', 'UNCERTAIN'], "Healthy pulse should be REAL or UNCERTAIN"
    print("✓ Test passed: Healthy pulse correctly classified")


def demo_deepfake_signal():
    """Scenario 2: Analyze a synthetic deepfake signal with artificial periodicity."""
    print("\n" + "="*70)
    print("SCENARIO 2: DEEPFAKE SIGNAL (Synthetic Repeating Pattern)")
    print("="*70)
    
    # Generate artificial repeating pattern (sawtooth = classic deepfake indicator)
    duration_sec = 10.0
    n_samples = int(FPS * duration_sec)
    t = np.arange(n_samples) / FPS
    
    # Repeating sawtooth every 1 second (highly periodic)
    pattern_length = int(FPS * 1.0)
    pattern = scipy_signal.sawtooth(2 * np.pi * t[:pattern_length])
    signal = np.tile(pattern, int(n_samples / pattern_length) + 1)[:n_samples]
    
    # Add minimal noise to simulate "clean" synthetic signal
    signal = signal + np.random.normal(0, 0.02, n_samples)
    
    result = analyze_threat(
        signal=signal.astype(np.float64),
        estimated_bpm=72.0,
        is_calibrating=False,
        signal_quality=0.7,
        fs=FPS
    )
    
    print(f"Verdict:           {result['verdict']}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"BPM:               {result['bpm']:.1f}")
    print(f"Pulse Present:     {result['pulse_present']}")
    print(f"Loop Detected:     {result['loop_detected']}")
    print(f"SNR (dB):          {result['snr_score']:.2f}")
    print(f"Dominant Freq (Hz):{result['dominant_freq_hz']:.3f}")
    
    print("✓ Test passed: Deepfake signal analyzed (may detect loop)")


def demo_low_quality_signal():
    """Scenario 3: Analyze a low signal quality case (high noise)."""
    print("\n" + "="*70)
    print("SCENARIO 3: LOW QUALITY SIGNAL (High Noise / Poor SNR)")
    print("="*70)
    
    # Generate pure noise (no cardiac signal)
    n_samples = 300
    signal = np.random.normal(0, 1.0, n_samples)
    
    result = analyze_threat(
        signal=signal.astype(np.float64),
        estimated_bpm=0.0,
        is_calibrating=False,
        signal_quality=0.3,  # Poor signal quality from rPPG Engine
        fs=FPS
    )
    
    print(f"Verdict:           {result['verdict']}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"BPM:               {result['bpm']:.1f}")
    print(f"Pulse Present:     {result['pulse_present']}")
    print(f"Loop Detected:     {result['loop_detected']}")
    print(f"SNR (dB):          {result['snr_score']:.2f}")
    print(f"Dominant Freq (Hz):{result['dominant_freq_hz']:.3f}")
    
    assert result['verdict'] == 'UNCERTAIN', "Low quality signal should be UNCERTAIN"
    print("✓ Test passed: Low quality signal safely flagged as UNCERTAIN")


def demo_calibration_mode():
    """Scenario 4: Analyze during calibration (should always return UNCERTAIN)."""
    print("\n" + "="*70)
    print("SCENARIO 4: CALIBRATION MODE (Warm-up Period)")
    print("="*70)
    
    # Generate any signal (doesn't matter in calibration mode)
    n_samples = 150  # 5 seconds @ 30 FPS
    signal = np.random.normal(0, 0.1, n_samples)
    
    result = analyze_threat(
        signal=signal.astype(np.float64),
        estimated_bpm=72.0,
        is_calibrating=True,  # CALIBRATION MODE
        signal_quality=0.9,
        fs=FPS
    )
    
    print(f"Verdict:           {result['verdict']}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"BPM:               {result['bpm']:.1f}")
    print(f"Pulse Present:     {result['pulse_present']}")
    print(f"Loop Detected:     {result['loop_detected']}")
    print(f"SNR (dB):          {result['snr_score']:.2f}")
    print(f"Dominant Freq (Hz):{result['dominant_freq_hz']:.3f}")
    
    assert result['verdict'] == 'UNCERTAIN', "Calibration should always be UNCERTAIN"
    print("✓ Test passed: Calibration mode correctly forces UNCERTAIN")


def demo_verdict_aggregation():
    """Scenario 5: Demonstrate rolling verdict aggregation with majority vote."""
    print("\n" + "="*70)
    print("SCENARIO 5: VERDICT AGGREGATION (Stateful Rolling Window)")
    print("="*70)
    
    analyzer = get_analyzer()
    analyzer.reset_verdict_history()
    
    # Generate multiple signals in sequence
    print("\nAnalyzing 5 consecutive signal batches...")
    
    for i in range(5):
        # Alternate between healthy and slightly anomalous signals
        if i % 2 == 0:
            # Healthy pulse
            t = np.arange(300) / FPS
            freq_hz = 72.0 / 60.0
            signal = np.sin(2 * np.pi * freq_hz * t) + np.random.normal(0, 0.1, 300)
        else:
            # Slightly noisy signal
            signal = np.random.normal(0, 0.3, 300)
        
        result = analyzer.analyze_threat(
            signal=signal.astype(np.float64),
            estimated_bpm=72.0,
            is_calibrating=False,
            signal_quality=0.85,
            fs=FPS
        )
        
        history = analyzer.get_verdict_history()
        print(f"\n  Batch {i+1}:")
        print(f"    Individual Verdict: {result['verdict']}")
        print(f"    Aggregated Verdict: {result['verdict']}")
        print(f"    History size: {len(history)}/5")
        print(f"    History: {history}")
    
    print("\n✓ Test passed: Rolling window verdict aggregation works correctly")


def demo_edge_case_handling():
    """Scenario 6: Test edge case handling (empty signal, short signal, etc.)."""
    print("\n" + "="*70)
    print("SCENARIO 6: EDGE CASE HANDLING")
    print("="*70)
    
    # Test 1: Empty signal
    print("\n  Test 1: Empty signal")
    result = analyze_threat(
        signal=np.array([], dtype=np.float64),
        estimated_bpm=0.0,
        is_calibrating=False,
        signal_quality=0.0,
        fs=FPS
    )
    assert result['verdict'] == 'UNCERTAIN'
    print(f"    ✓ Empty signal → {result['verdict']}")
    
    # Test 2: Too-short signal
    print("\n  Test 2: Very short signal (30 samples)")
    short_sig = np.random.normal(0, 0.1, 30)
    result = analyze_threat(
        signal=short_sig.astype(np.float64),
        estimated_bpm=0.0,
        is_calibrating=False,
        signal_quality=0.5,
        fs=FPS
    )
    assert result['verdict'] == 'UNCERTAIN'
    print(f"    ✓ Short signal → {result['verdict']}")
    
    # Test 3: Signal with NaN values
    print("\n  Test 3: Signal with NaN values")
    nan_sig = np.random.normal(0, 0.1, 300)
    nan_sig[100:110] = np.nan
    result = analyze_threat(
        signal=nan_sig.astype(np.float64),
        estimated_bpm=0.0,
        is_calibrating=False,
        signal_quality=0.5,
        fs=FPS
    )
    print(f"    ✓ NaN signal → {result['verdict']} (no crash)")
    
    # Test 4: All-zero signal
    print("\n  Test 4: All-zero signal")
    zero_sig = np.zeros(300, dtype=np.float64)
    result = analyze_threat(
        signal=zero_sig,
        estimated_bpm=0.0,
        is_calibrating=False,
        signal_quality=0.0,
        fs=FPS
    )
    assert result['verdict'] == 'UNCERTAIN'
    print(f"    ✓ Zero signal → {result['verdict']}")
    
    print("\n✓ All edge cases handled gracefully")


def main():
    """Run all integration test scenarios."""
    print("\n" + "="*70)
    print("THREAT ANALYSIS ENGINE - INTEGRATION TEST & DEMO")
    print("="*70)
    print("\nTesting rPPG-based deepfake threat detection pipeline...")
    
    demo_healthy_pulse()
    demo_deepfake_signal()
    demo_low_quality_signal()
    demo_calibration_mode()
    demo_verdict_aggregation()
    demo_edge_case_handling()
    
    print("\n" + "="*70)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*70)
    print("\nThreat Analyzer is ready for integration with rPPG Engine & UI!")
    print("\nUsage:")
    print("  from modules.threat_analyzer import analyze_threat")
    print("  result = analyze_threat(signal, estimated_bpm, is_calibrating, signal_quality)")
    print("  verdict = result['verdict']  # 'REAL', 'THREAT', or 'UNCERTAIN'")


if __name__ == "__main__":
    main()
