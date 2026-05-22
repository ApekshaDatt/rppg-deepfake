"""
Threat Analysis Engine Configuration Constants
Centralized configuration for all thresholds, parameters, and output formatting.
"""

# ============================================================================
# CORE THRESHOLDS
# ============================================================================
BANDPASS_LOW = 0.7              # Hz (42 BPM) - lower cardiac frequency bound
BANDPASS_HIGH = 3.0             # Hz (180 BPM) - upper cardiac frequency bound
THREAT_THRESHOLD = 0.75         # threat_score >= 0.75 → THREAT verdict
LOOP_CORR_THRESHOLD = 0.92      # autocorrelation > 0.92 → periodic loop detected
SNR_THRESHOLD = 5.0             # SNR >= 5.0 dB is considered good signal quality
VERDICT_WINDOW = 5              # rolling window size for verdict aggregation
FPS = 30                         # frames per second (video capture rate)
BUFFER_SIZE = 300               # samples (10 sec @ 30 FPS)
CALIBRATION_FRAMES = 150        # 5 sec @ 30 FPS = warm-up period

# ============================================================================
# DERIVED CONSTANTS
# ============================================================================
FILTER_ORDER = 4                # Butterworth IIR filter order
MIN_SIGNAL_LENGTH = 60          # minimum samples required for reliable FFT/analysis
MIN_AUTOCORR_LENGTH = 20        # minimum samples for loop detection
BPM_MIN = 40                     # physiological lower bound (BPM)
BPM_MAX = 200                    # physiological upper bound (BPM)

# ============================================================================
# OUTPUT FORMATTING (ROUNDING DECIMALS)
# ============================================================================
CONFIDENCE_DECIMALS = 2         # e.g., 0.92
BPM_DECIMALS = 1                # e.g., 72.5
SNR_DECIMALS = 2                # e.g., 8.34
FREQ_DECIMALS = 3               # e.g., 1.234 Hz

# ============================================================================
# COMPONENT WEIGHTS (must sum to 1.0)
# ============================================================================
PULSE_PRESENCE_WEIGHT = 0.20    # pulse detected (yes/no)
LOOP_SCORE_WEIGHT = 0.40        # periodic loop correlation (most reliable deepfake signal)
SNR_WEIGHT = 0.30               # signal-to-noise ratio
SIGNAL_QUALITY_WEIGHT = 0.10    # upstream signal quality from Person B

# ============================================================================
# VERDICT CATEGORIES
# ============================================================================
VERDICT_REAL = "REAL"           # authentic pulse (green, no alarm)
VERDICT_THREAT = "THREAT"       # deepfake detected (red + alarm)
VERDICT_UNCERTAIN = "UNCERTAIN" # insufficient data / calibration (yellow, safe)

# ============================================================================
# THREAT SCORE BOUNDARIES
# ============================================================================
THREAT_SCORE_REAL_BOUNDARY = 0.40      # threat_score <= 0.40 → REAL
THREAT_SCORE_UNCERTAIN_LOW = 0.40      # 0.40 < threat_score < 0.75 → UNCERTAIN
THREAT_SCORE_UNCERTAIN_HIGH = 0.75     # (same as THREAT_THRESHOLD)

# ============================================================================
# CONFIDENCE COMPUTATION PARAMETERS
# ============================================================================
BASE_CONFIDENCE = 0.5           # starting confidence value
CONF_BONUS_HIGH_SNR = 0.3       # bonus if SNR > SNR_THRESHOLD
CONF_BONUS_CLEAR_SCORE = 0.15   # bonus if threat_score far from threshold
CONF_BONUS_PULSE_PRESENT = 0.05 # bonus if pulse detected
CONF_BONUS_GOOD_QUALITY = 0.10  # bonus if signal_quality > 0.8
SIGNAL_QUALITY_THRESHOLD = 0.8  # threshold for good signal quality bonus
