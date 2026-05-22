import numpy as np
from config import FPS

class RPPGAlgorithms:
    @staticmethod
    def green_method(rgb_signals):
        return rgb_signals[:, 1]

    @staticmethod
    def chrom_method(rgb_signals):
        if len(rgb_signals) < 2:
            return rgb_signals[:, 1]
        
        means = np.mean(rgb_signals, axis=0)
        norm_rgb = rgb_signals / (means + 1e-6)
        
        b, g, r = norm_rgb[:, 0], norm_rgb[:, 1], norm_rgb[:, 2]
        
        x = 3 * r - 2 * g
        y = 1.5 * r + g - 1.5 * b
        
        # SCIENTIFIC FIX: According to de Haan & Jeanne (2013), X and Y must be
        # bandpass filtered BEFORE computing their standard deviations (alpha ratio).
        # Otherwise, low-frequency head motion completely dominates the variance,
        # ruining the pulse extraction and causing false high-frequency noise peaks.
        try:
            from modules.threat_analyzer.fft_analyzer import apply_bandpass
            xf = apply_bandpass(x, fps=FPS, low=0.7, high=3.0)
            yf = apply_bandpass(y, fps=FPS, low=0.7, high=3.0)
            
            # Fallback to unfiltered if signal is too short for the filter order
            if len(xf) == 0 or len(yf) == 0:
                xf, yf = x, y
        except Exception:
            xf, yf = x, y
        
        sx = np.std(xf)
        sy = np.std(yf)
        
        if sy == 0:
            return x
        
        alpha = sx / sy
        return x - alpha * y

    @staticmethod
    def estimate_bpm(signal, fps=FPS):
        if len(signal) < 2:
            return 0.0
        
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1/fps)
        fft_vals = np.abs(np.fft.rfft(signal))
        
        mask = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(mask):
            return 0.0
            
        masked_freqs = freqs[mask]
        masked_fft = fft_vals[mask]
        
        peak_freq = masked_freqs[np.argmax(masked_fft)]
        return peak_freq * 60.0
