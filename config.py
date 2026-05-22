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
SNR_THRESHOLD = 8.0             # Strict enough to reject noise, low enough for webcams
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

# ============================================================================
# FACE DETECTION PARAMETERS (Person A)
# ============================================================================
ROI_PADDING = 10                        # pixels to shrink ROI inward from face edges
MIN_FACE_SIZE = (80, 80)                # minimum (w, h) for a valid detected face

HAAR_SCALE_FACTOR = 1.1                 # Haar cascade scale factor (lower = more sensitive)
HAAR_MIN_NEIGHBORS = 5                  # Haar minimum neighbors (lower = more detections)

# ============================================================================
# ROI GEOMETRY RATIOS (Person A)
# ============================================================================
ROI_FOREHEAD_HEIGHT_RATIO = 0.30        # forehead = top 30% of face height
ROI_FOREHEAD_WIDTH_RATIO = 0.60         # forehead = center 60% of face width
ROI_CHEEK_HEIGHT_RATIO = 0.40           # cheeks = middle 40% of face height
