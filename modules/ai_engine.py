import numpy as np

class AIEngineMock:
    def __init__(self):
        self.risk_history = []
        
    def analyze_signal(self, filtered_signal):
        """
        Analyzes the rPPG signal to determine Authenticity Risk.
        Real biological signals have a strong periodicity (high SNR).
        Deepfakes often lose this subtle signal, resulting in noise.
        """
        if not filtered_signal or len(filtered_signal) < 30:
            return 50.0, 50.0  # Default uncertain state
            
        signal_arr = np.array(filtered_signal)
        
        # Calculate Signal-to-Noise Ratio (SNR) proxy
        # Variance of the signal vs mean absolute deviation
        variance = np.var(signal_arr)
        
        # A good human pulse has a clear, rhythmic variance. 
        # Deepfake noise is usually flatter or entirely random (low variance or extremely high erratic variance).
        # We will map variance to Signal Integrity (0 to 100)
        
        # Typical normalized filtered rPPG variance might be around 0.5 to 5.0
        integrity = min(100.0, max(0.0, (variance * 10.0) + 40.0))
        
        # If the signal is too flat, risk is very high (deepfake artifact)
        if variance < 0.1:
            risk = 85.0 + np.random.uniform(0, 10)
            integrity = 10.0 + np.random.uniform(0, 10)
        else:
            # Good signal = low risk
            risk = max(0.0, 100.0 - integrity) + np.random.uniform(-5, 5)
            
        # Smooth the risk
        self.risk_history.append(risk)
        if len(self.risk_history) > 10:
            self.risk_history.pop(0)
            
        smoothed_risk = sum(self.risk_history) / len(self.risk_history)
        
        # Ensure bounds
        smoothed_risk = min(100.0, max(0.0, smoothed_risk))
        integrity = min(100.0, max(0.0, integrity))
        
        return smoothed_risk, integrity
