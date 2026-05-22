import numpy as np
import pytest
from modules.rppg_engine.preprocessor import Preprocessor
from modules.rppg_engine.rppg_algorithms import RPPGAlgorithms
from config import FPS, BUFFER_SIZE

def test_bpm_estimation():
    t = np.linspace(0, BUFFER_SIZE / FPS, BUFFER_SIZE)
    freq = 1.2
    pure_signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.1, BUFFER_SIZE)
    raw_signal = pure_signal + noise
    
    processed_signal = Preprocessor.process(raw_signal)
    bpm = RPPGAlgorithms.estimate_bpm(processed_signal, fps=FPS)
    
    assert abs(bpm - 72) <= 5

if __name__ == "__main__":
    test_bpm_estimation()
    print("Test passed!")
