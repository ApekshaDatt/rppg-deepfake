import numpy as np
from config import BUFFER_SIZE

class SignalExtractor:
    def __init__(self):
        self.buffer = np.zeros((BUFFER_SIZE, 3))
        self.index = 0
        self.count = 0

    def extract_rgb_means(self, roi):
        if roi is None or roi.size == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(roi, axis=(0, 1))

    def update_buffer(self, rgb_values):
        self.buffer[self.index] = rgb_values
        self.index = (self.index + 1) % BUFFER_SIZE
        if self.count < BUFFER_SIZE:
            self.count += 1

    def get_signals(self):
        if self.count < BUFFER_SIZE:
            return self.buffer[:self.count]
        return np.roll(self.buffer, -self.index, axis=0)
