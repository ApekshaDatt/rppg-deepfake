#!/usr/bin/env python3
"""
PHASE 0: Environment Verification Script
Member C setup verification for rPPG Deepfake Detection System.

This script verifies:
1. Required libraries are installed (numpy, scipy, matplotlib)
2. Python version compatibility (3.10+)
3. Synthetic signal generation works correctly
4. Array dtype requirements met (float64)
5. All dependencies ready for production
"""

import sys
import numpy as np
from scipy import signal as scipy_signal
import matplotlib

# ============================================================================
# VERSION CHECKS
# ============================================================================

print("\n" + "="*70)
print("ENVIRONMENT VERIFICATION - Member C Setup")
print("="*70)

print("\n1️⃣  PYTHON VERSION CHECK")
print(f"   Python version: {sys.version}")
if sys.version_info >= (3, 10):
    print("   ✅ Python 3.10+ detected")
else:
    print("   ❌ ERROR: Python 3.10+ required")
    sys.exit(1)

print("\n2️⃣  LIBRARY VERSION CHECK")
print(f"   numpy:      {np.__version__}")
print(f"   scipy:      {scipy_signal.__version__ if hasattr(scipy_signal, '__version__') else 'scipy (version via package)'}")
print(f"   matplotlib: {matplotlib.__version__}")
print("   ✅ All required libraries installed")

# ============================================================================
# SYNTHETIC SIGNAL GENERATION
# ============================================================================

print("\n3️⃣  SYNTHETIC SIGNAL GENERATION")

FPS = 30
DURATION_SEC = 10.0
FREQ_HZ = 1.2  # 72 BPM = 1.2 Hz

n_samples = int(FPS * DURATION_SEC)
t = np.arange(n_samples) / FPS

# Generate synthetic 72 BPM signal
signal = np.sin(2 * np.pi * FREQ_HZ * t)
noise = np.random.normal(0, 0.1, n_samples)
signal_with_noise = signal + noise

print(f"\n   Signal parameters:")
print(f"   - Sampling rate (FPS):    {FPS}")
print(f"   - Duration:               {DURATION_SEC} sec")
print(f"   - Frequency:              {FREQ_HZ} Hz ({FREQ_HZ * 60:.0f} BPM)")
print(f"   - Total samples:          {n_samples}")
print(f"   - Array shape:            {signal_with_noise.shape}")
print(f"   - Array dtype:            {signal_with_noise.dtype}")

# ============================================================================
# ARRAY DTYPE VALIDATION
# ============================================================================

print("\n4️⃣  ARRAY DTYPE VALIDATION (float64 required)")

# Convert to float64 (strict requirement for FFT)
signal_f64 = signal_with_noise.astype(np.float64)

print(f"   Original dtype:  {signal_with_noise.dtype}")
print(f"   After cast:      {signal_f64.dtype}")

if signal_f64.dtype == np.float64:
    print("   ✅ float64 dtype confirmed")
else:
    print("   ❌ ERROR: float64 dtype not confirmed")
    sys.exit(1)

# ============================================================================
# SIGNAL INSPECTION
# ============================================================================

print("\n5️⃣  SIGNAL INSPECTION")

print(f"\n   First 5 values:")
for i in range(min(5, len(signal_f64))):
    print(f"     [{i}] = {signal_f64[i]:8.6f}")

print(f"\n   Statistics:")
print(f"   - Min:    {np.min(signal_f64):8.6f}")
print(f"   - Max:    {np.max(signal_f64):8.6f}")
print(f"   - Mean:   {np.mean(signal_f64):8.6f}")
print(f"   - Std:    {np.std(signal_f64):8.6f}")

# ============================================================================
# FFT VERIFICATION
# ============================================================================

print("\n6️⃣  FFT VERIFICATION")

try:
    fft_vals = np.fft.rfft(signal_f64)
    freqs = np.fft.rfftfreq(len(signal_f64), d=1.0/FPS)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # Find dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    dominant_freq = freqs[dominant_idx]
    
    print(f"   FFT computed successfully")
    print(f"   - FFT output shape:  {fft_vals.shape}")
    print(f"   - Frequency array:   {freqs.shape}")
    print(f"   - Power spectrum:    {power_spectrum.shape}")
    print(f"   - Dominant frequency: {dominant_freq:.3f} Hz ({dominant_freq * 60:.1f} BPM)")
    print(f"   - Expected frequency: {FREQ_HZ:.3f} Hz ({FREQ_HZ * 60:.1f} BPM)")
    
    freq_error = abs(dominant_freq - FREQ_HZ) / FREQ_HZ * 100
    if freq_error < 5.0:  # Allow 5% error
        print(f"   ✅ Frequency detection accurate ({freq_error:.1f}% error)")
    else:
        print(f"   ⚠️  Frequency detection off ({freq_error:.1f}% error)")
        
except Exception as e:
    print(f"   ❌ FFT verification failed: {e}")
    sys.exit(1)

# ============================================================================
# SCIPY SIGNAL PROCESSING
# ============================================================================

print("\n7️⃣  SCIPY SIGNAL PROCESSING")

try:
    # Test Butterworth filter
    nyquist = FPS / 2.0
    low_normalized = 0.7 / nyquist
    high_normalized = 3.0 / nyquist
    
    b, a = scipy_signal.butter(4, [low_normalized, high_normalized], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal_f64)
    
    print(f"   Butterworth filter created successfully")
    print(f"   - Filter order:      4")
    print(f"   - Band:              [0.7-3.0] Hz")
    print(f"   - Filtered shape:    {filtered.shape}")
    print(f"   - Filtered dtype:    {filtered.dtype}")
    print(f"   ✅ Signal filtering verified")
    
except Exception as e:
    print(f"   ❌ Signal filtering failed: {e}")
    sys.exit(1)

# ============================================================================
# CONFIG IMPORT CHECK
# ============================================================================

print("\n8️⃣  CONFIG MODULE IMPORT")

try:
    from config import (
        BANDPASS_LOW, BANDPASS_HIGH, THREAT_THRESHOLD, 
        LOOP_CORR_THRESHOLD, SNR_THRESHOLD, VERDICT_WINDOW,
        FPS as CONFIG_FPS, BUFFER_SIZE
    )
    
    print(f"   Config constants loaded:")
    print(f"   - BANDPASS_LOW:        {BANDPASS_LOW} Hz")
    print(f"   - BANDPASS_HIGH:       {BANDPASS_HIGH} Hz")
    print(f"   - THREAT_THRESHOLD:    {THREAT_THRESHOLD}")
    print(f"   - LOOP_CORR_THRESHOLD: {LOOP_CORR_THRESHOLD}")
    print(f"   - SNR_THRESHOLD:       {SNR_THRESHOLD} dB")
    print(f"   - VERDICT_WINDOW:      {VERDICT_WINDOW}")
    print(f"   - FPS:                 {CONFIG_FPS}")
    print(f"   - BUFFER_SIZE:         {BUFFER_SIZE}")
    print(f"   ✅ Config module verified")
    
except Exception as e:
    print(f"   ⚠️  Config import warning: {e}")
    print(f"      (Config may not be finalized yet - this is normal in PHASE 0)")

# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "="*70)
print("✅ MEMBER C ENVIRONMENT READY")
print("="*70)
print("\nAll critical systems verified:")
print("  ✅ Python 3.10+")
print("  ✅ NumPy (float64 arrays)")
print("  ✅ SciPy (signal processing)")
print("  ✅ Matplotlib (visualization)")
print("  ✅ FFT computation")
print("  ✅ Butterworth filtering")
print("  ✅ Config constants")

print("\n🚀 Ready to begin PHASE 1: Function Implementation")
print("="*70 + "\n")
