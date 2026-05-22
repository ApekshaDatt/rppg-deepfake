import numpy as np
from scipy import signal

class RPPGExtractor:
    def __init__(self, fps=30.0, buffer_size=150):
        self.fps = fps
        self.buffer_size = buffer_size
        
        # We will track the green channel since it absorbs hemoglobin light best
        self.raw_signal = []
        self.filtered_signal = []
        self.timestamps = []
        self.current_bpm = 0.0
        
    def add_frame_data(self, mean_green_value, timestamp):
        """Adds a new data point to the buffer."""
        self.raw_signal.append(mean_green_value)
        self.timestamps.append(timestamp)
        
        if len(self.raw_signal) > self.buffer_size:
            self.raw_signal.pop(0)
            self.timestamps.pop(0)
            
        self._process_signal()
        
    def _process_signal(self):
        """Applies bandpass filter and calculates BPM."""
        if len(self.raw_signal) < self.fps * 3: # Need at least 3 seconds of data
            self.filtered_signal = np.zeros(len(self.raw_signal)).tolist()
            return

        # Detrend the signal
        detrended = signal.detrend(self.raw_signal)
        
        # Bandpass filter (0.7 Hz to 3.0 Hz corresponds to 42 - 180 BPM)
        nyquist = 0.5 * self.fps
        low = 0.7 / nyquist
        high = 3.0 / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        
        filtered = signal.filtfilt(b, a, detrended)
        self.filtered_signal = filtered.tolist()
        
        # Calculate BPM using FFT
        N = len(filtered)
        freqs = np.fft.rfftfreq(N, d=1.0/self.fps)
        fft_mags = np.abs(np.fft.rfft(filtered))
        
        # Ignore frequencies outside our human range
        valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]
        if len(valid_idx) > 0:
            peak_idx = valid_idx[np.argmax(fft_mags[valid_idx])]
            peak_freq = freqs[peak_idx]
            self.current_bpm = peak_freq * 60.0
