"""
FFT Analyzer: Frequency-domain signal processing for rPPG threat analysis.
Extracts spectral features (dominant frequency, SNR, BPM) from time-series signals.
"""

import numpy as np
from scipy import signal as scipy_signal
from config import (
    BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, MIN_SIGNAL_LENGTH,
    SNR_THRESHOLD, BPM_MIN, BPM_MAX
)


def bandpass_filter(signal: np.ndarray, low_hz: float, high_hz: float, fs: int) -> np.ndarray:
    """Apply Butterworth bandpass filter to extract cardiac frequency band."""
    # Guard: empty or too-short signal
    if signal.size == 0:
        return np.array([])
    if signal.size < FILTER_ORDER * 4:
        # Signal too short to filter reliably; return as-is (with mean padding conceptually)
        return signal.astype(float)
    
    # Guard: all-zero signal
    if np.allclose(signal, 0.0):
        return signal.astype(float)
    
    try:
        # Design Butterworth bandpass filter
        nyquist_freq = fs / 2.0
        low_normalized = low_hz / nyquist_freq
        high_normalized = high_hz / nyquist_freq
        
        # Clamp normalized frequencies to valid range (0, 1)
        low_normalized = np.clip(low_normalized, 0.01, 0.99)
        high_normalized = np.clip(high_normalized, 0.01, 0.99)
        
        # Ensure low < high
        if low_normalized >= high_normalized:
            low_normalized = max(0.01, high_normalized - 0.1)
        
        # Create filter
        b, a = scipy_signal.butter(FILTER_ORDER, [low_normalized, high_normalized], btype='band')
        
        # Apply filter (forward-backward to preserve phase)
        filtered = scipy_signal.filtfilt(b, a, signal)
        return filtered.astype(np.float64)
    
    except Exception as e:
        # On any filtering error, return signal as-is
        print(f"Warning: bandpass_filter failed: {e}. Returning unfiltered signal.")
        return signal.astype(np.float64)


def compute_snr(signal: np.ndarray, low_hz: float, high_hz: float, fs: int) -> float:
    """Compute Signal-to-Noise Ratio in the cardiac frequency band (dB)."""
    # Guard: insufficient signal length
    if signal.size < MIN_SIGNAL_LENGTH:
        return 0.0
    
    # Guard: all-zero signal
    if np.allclose(signal, 0.0):
        return 0.0
    
    try:
        # Compute FFT
        n_fft = len(signal)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
        fft_vals = np.fft.rfft(signal)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Define signal band: [low_hz, high_hz]
        signal_band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        signal_power = np.sum(power_spectrum[signal_band_mask])
        
        # Define noise band: [0, low_hz) + (high_hz, fs/2]
        noise_band_mask = (freqs < low_hz) | (freqs > high_hz)
        noise_power = np.sum(power_spectrum[noise_band_mask])
        
        # Compute SNR in dB
        if noise_power < 1e-10:  # Avoid log(0)
            return 100.0  # Return capped value (unrealistic but safe)
        if signal_power < 1e-10:
            return 0.0
        
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return float(np.clip(snr_db, -50.0, 100.0))  # Clamp to reasonable range
    
    except Exception as e:
        print(f"Warning: compute_snr failed: {e}")
        return 0.0


def extract_dominant_frequency(signal: np.ndarray, fs: int) -> tuple[float, float]:
    """Extract dominant frequency in cardiac band via FFT. Returns (frequency_hz, power_magnitude)."""
    # Guard: insufficient signal length
    if signal.size < MIN_SIGNAL_LENGTH:
        return (0.0, 0.0)
    
    # Guard: all-zero signal
    if np.allclose(signal, 0.0):
        return (0.0, 0.0)
    
    try:
        # Compute FFT
        n_fft = len(signal)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
        fft_vals = np.fft.rfft(signal)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Search for max power in cardiac band [BANDPASS_LOW, BANDPASS_HIGH]
        cardiac_band_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
        
        if not np.any(cardiac_band_mask):
            # No energy in cardiac band
            return (0.0, 0.0)
        
        # Find frequency with max power in cardiac band
        cardiac_power = power_spectrum[cardiac_band_mask]
        max_idx_in_band = np.argmax(cardiac_power)
        max_idx_global = np.where(cardiac_band_mask)[0][max_idx_in_band]
        
        dominant_freq = freqs[max_idx_global]
        dominant_power = power_spectrum[max_idx_global]
        
        return (float(dominant_freq), float(dominant_power))
    
    except Exception as e:
        print(f"Warning: extract_dominant_frequency failed: {e}")
        return (0.0, 0.0)


def estimate_bpm_from_freq(freq_hz: float) -> float:
    """Convert frequency (Hz) to BPM. Validate against physiological range [40, 200]."""
    # Convert Hz to BPM
    bpm = freq_hz * 60.0
    
    # Validate range
    if bpm < BPM_MIN or bpm > BPM_MAX:
        return 0.0  # Invalid BPM
    
    return float(bpm)
