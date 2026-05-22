import numpy as np
from .signal_extractor import SignalExtractor
from .preprocessor import Preprocessor
from .rppg_algorithms import RPPGAlgorithms
from config import BUFFER_SIZE, FPS, CALIBRATION_FRAMES

class RPPGEngine:
    def __init__(self):
        self.extractor = SignalExtractor()
        self.preprocessor = Preprocessor()
        self.algorithms = RPPGAlgorithms()
        self.method = "CHROM"

    def __call__(self, roi_forehead, roi_cheeks, frame_count):
        means_f = self.extractor.extract_rgb_means(roi_forehead) if roi_forehead is not None else None
        means_c = self.extractor.extract_rgb_means(roi_cheeks) if roi_cheeks is not None else None
        
        if means_f is not None and means_c is not None:
            rgb_vals = (means_f + means_c) / 2
        elif means_f is not None:
            rgb_vals = means_f
        elif means_c is not None:
            rgb_vals = means_c
        else:
            rgb_vals = np.array([0.0, 0.0, 0.0])
            
        self.extractor.update_buffer(rgb_vals)
        raw_signals = self.extractor.get_signals()
        
        if self.method == "CHROM":
            pulse_signal = self.algorithms.chrom_method(raw_signals)
        else:
            pulse_signal = self.algorithms.green_method(raw_signals)
            
        processed_signal = self.preprocessor.process(pulse_signal)
        bpm = self.algorithms.estimate_bpm(processed_signal)
        
        is_calibrating = frame_count < CALIBRATION_FRAMES
        quality = self._calculate_quality(processed_signal)
        
        return {
            "rppg_signal": processed_signal,
            "estimated_bpm": float(bpm),
            "is_calibrating": is_calibrating,
            "signal_quality": float(quality),
            "method_used": self.method
        }

    def _calculate_quality(self, signal):
        if len(signal) < 2:
            return 0.0
        fft_vals = np.abs(np.fft.rfft(signal))
        peak_idx = np.argmax(fft_vals)
        if peak_idx == 0:
            return 0.0
        peak_val = fft_vals[peak_idx]
        total_val = np.sum(fft_vals)
        return peak_val / total_val if total_val > 0 else 0.0

engine = RPPGEngine()

def process_signal(roi_forehead, roi_cheeks, frame_count, buffer_size=BUFFER_SIZE):
    return engine(roi_forehead, roi_cheeks, frame_count)

__all__ = ["process_signal", "RPPGEngine"]
