import numpy as np
from scipy import signal

class Preprocessor:
    @staticmethod
    def process(raw_signal):
        if len(raw_signal) < 2:
            return raw_signal
        detrended = signal.detrend(raw_signal)
        std = np.std(detrended)
        if std == 0:
            return detrended - np.mean(detrended)
        return (detrended - np.mean(detrended)) / std
