import numpy as np
from scipy import signal

# Minimum absolute signal variance before normalization.
# If std is below this threshold the signal carries no meaningful information
# (it's float machine-epsilon noise from a constant/near-constant input).
# Dividing by a sub-threshold std would amplify noise by ~1e16, creating
# a fake signal that fools the FFT. We return zeros instead.
_MIN_SIGNAL_STD = 1e-6

class Preprocessor:
    @staticmethod
    def process(raw_signal):
        if len(raw_signal) < 2:
            return raw_signal
        detrended = signal.detrend(raw_signal)
        std = np.std(detrended)
        # Use a meaningful epsilon, not exact zero, to catch float noise amplification
        if std < _MIN_SIGNAL_STD:
            return np.zeros(len(detrended), dtype=np.float64)
        return (detrended - np.mean(detrended)) / std
