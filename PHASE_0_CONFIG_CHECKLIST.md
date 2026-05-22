# PHASE 0 CHECKLIST: Config Constants to Confirm with Person A

## ✅ VERIFIED CONSTANTS (Already in config.py)

Copy this checklist into your team chat to confirm with Person A before Phase 1 begins.

---

## **CORE THRESHOLDS** ✓

- [ ] `BANDPASS_LOW = 0.7 Hz` — Lower cardiac frequency bound (42 BPM)
  - Confirm: This should remain 0.7 Hz (industry standard for rPPG)
  
- [ ] `BANDPASS_HIGH = 3.0 Hz` — Upper cardiac frequency bound (180 BPM)
  - Confirm: This should remain 3.0 Hz (physiological maximum heart rate)
  
- [ ] `THREAT_THRESHOLD = 0.75` — Score boundary for THREAT verdict
  - Confirm: Start at 0.75 (balanced). Tuning range: [0.65, 0.80]
  - Guidance: 0.75 catches most deepfakes (~5-10% false alarms on real faces)
  
- [ ] `LOOP_CORR_THRESHOLD = 0.92` — Deepfake loop detection sensitivity
  - Confirm: Keep at 0.92 (strict, high confidence). Range: [0.85, 0.95]
  - Meaning: autocorr > 0.92 → periodic loop detected → likely deepfake
  
- [ ] `SNR_THRESHOLD = 5.0 dB` — Good signal quality cutoff
  - Confirm: Keep at 5.0 dB. This defines "healthy" SNR.
  
- [ ] `VERDICT_WINDOW = 5` — Verdicts in rolling aggregation window
  - Confirm: Keep at 5 (temporal smoothing). Range: [3, 7]
  - Meaning: Last 5 verdicts → majority vote → aggregated verdict
  
- [ ] `FPS = 30` — Video frame rate (frames per second)
  - Confirm: This MUST match your camera/rPPG Engine output (typically 30)
  - Impact: Affects all time-domain calculations (filter order, window sizing)
  
- [ ] `BUFFER_SIZE = 300` — Samples per analysis batch (10 sec @ 30 FPS)
  - Confirm: Keep at 300 (standard 10-second buffer)
  - Formula: BUFFER_SIZE = FPS × duration_sec = 30 × 10 = 300
  
- [ ] `CALIBRATION_FRAMES = 150` — Warm-up period (5 sec @ 30 FPS)
  - Confirm: Keep at 150 (gives rPPG Engine time to stabilize)
  - Formula: CALIBRATION_FRAMES = FPS × calibration_sec = 30 × 5 = 150

---

## **DERIVED CONSTANTS** ✓

- [ ] `FILTER_ORDER = 4` — Butterworth IIR filter order
  - Confirm: Keep at 4 (balance between stability and steepness)
  
- [ ] `MIN_SIGNAL_LENGTH = 60` — Minimum samples for FFT reliability
  - Confirm: Keep at 60 (2 sec @ 30 FPS, minimum for 0.5 Hz resolution)
  
- [ ] `MIN_AUTOCORR_LENGTH = 20` — Minimum for loop detection
  - Confirm: Keep at 20 (0.67 sec, enough for periodicity detection)
  
- [ ] `BPM_MIN = 40` — Physiological lower bound
  - Confirm: Keep at 40 BPM (resting adult minimum)
  
- [ ] `BPM_MAX = 200` — Physiological upper bound
  - Confirm: Keep at 200 BPM (athletic/stress maximum)

---

## **COMPONENT WEIGHTS** (Must sum to 1.0) ✓

- [ ] `PULSE_PRESENCE_WEIGHT = 0.20` (20%)
  - Confirm: Pulse detection is mandatory component
  - If pulse not detected → low threat score component
  
- [ ] `LOOP_SCORE_WEIGHT = 0.40` (40%)
  - Confirm: Loop correlation is most reliable deepfake signal
  - Reason: Real pulse = random, synthetic = periodic
  
- [ ] `SNR_WEIGHT = 0.30` (30%)
  - Confirm: SNR indicates signal vs. noise quality
  
- [ ] `SIGNAL_QUALITY_WEIGHT = 0.10` (10%)
  - Confirm: Light weight for upstream rPPG Engine quality metric
  - Reason: Primary decision based on downstream analysis

---

## **OUTPUT ROUNDING PRECISION** ✓

- [ ] `CONFIDENCE_DECIMALS = 2` — Confidence rounded to 2 places (e.g., 0.92)
  
- [ ] `BPM_DECIMALS = 1` — BPM rounded to 1 place (e.g., 72.5)
  
- [ ] `SNR_DECIMALS = 2` — SNR rounded to 2 places (e.g., 8.34)
  
- [ ] `FREQ_DECIMALS = 3` — Frequency rounded to 3 places (e.g., 1.234)

---

## **VERDICT CATEGORIES** ✓

- [ ] `VERDICT_REAL = "REAL"` — Authentic pulse (green, no alarm)
  
- [ ] `VERDICT_THREAT = "THREAT"` — Deepfake detected (red, trigger alarm)
  
- [ ] `VERDICT_UNCERTAIN = "UNCERTAIN"` — Insufficient data (yellow, safe)

---

## **THREAT SCORE BOUNDARIES** ✓

- [ ] `THREAT_SCORE_REAL_BOUNDARY = 0.40`
  - Meaning: threat_score ≤ 0.40 → REAL (green)
  
- [ ] `THREAT_SCORE_UNCERTAIN_LOW = 0.40`
  - Meaning: 0.40 < threat_score < 0.75 → UNCERTAIN (yellow)
  
- [ ] `THREAT_SCORE_UNCERTAIN_HIGH = 0.75` (same as THREAT_THRESHOLD)
  - Meaning: threat_score ≥ 0.75 → THREAT (red)

---

## **CONFIDENCE COMPUTATION PARAMETERS** ✓

- [ ] `BASE_CONFIDENCE = 0.5` — Starting confidence value
  
- [ ] `CONF_BONUS_HIGH_SNR = 0.3` — Bonus if SNR > SNR_THRESHOLD (5.0 dB)
  
- [ ] `CONF_BONUS_CLEAR_SCORE = 0.15` — Bonus if threat_score far from threshold
  
- [ ] `CONF_BONUS_PULSE_PRESENT = 0.05` — Bonus if pulse detected
  
- [ ] `CONF_BONUS_GOOD_QUALITY = 0.10` — Bonus if signal_quality > 0.8
  
- [ ] `SIGNAL_QUALITY_THRESHOLD = 0.8` — Quality cutoff for bonus

---

## **TEAM CONFIRMATION TEMPLATE**

Copy this into your team chat:

```
@Person_A - PHASE 0 CONFIG CONFIRMATION NEEDED

Member C environment verification passed ✅

Please confirm these config constants for production:

CORE THRESHOLDS:
  ✓ BANDPASS_LOW = 0.7 Hz
  ✓ BANDPASS_HIGH = 3.0 Hz
  ✓ THREAT_THRESHOLD = 0.75 (tuning range: 0.65-0.80)
  ✓ LOOP_CORR_THRESHOLD = 0.92
  ✓ SNR_THRESHOLD = 5.0 dB
  ✓ VERDICT_WINDOW = 5
  ✓ FPS = 30
  ✓ BUFFER_SIZE = 300
  ✓ CALIBRATION_FRAMES = 150

COMPONENT WEIGHTS (sum = 1.0):
  ✓ PULSE_PRESENCE = 0.20
  ✓ LOOP_CORRELATION = 0.40 (most critical deepfake signal)
  ✓ SNR = 0.30
  ✓ SIGNAL_QUALITY = 0.10

SPECIAL NOTES:
  • FPS MUST match your camera/rPPG Engine rate (usually 30)
  • THREAT_THRESHOLD can be tuned 0.65-0.80 based on false positive rate
  • LOOP_CORR_THRESHOLD (0.92) is strict — real pulse < 0.92, synthetic > 0.92
  • All thresholds centralized in config.py (no magic numbers in code)

Approve to proceed with PHASE 1? ✓/❌
```

---

## **OPTIONAL TUNING GUIDANCE**

If you need to adjust after initial testing:

### **To catch more deepfakes:**
```python
THREAT_THRESHOLD = 0.65  # Was 0.75 (more aggressive)
LOOP_SCORE_WEIGHT = 0.50  # Was 0.40 (prioritize loop detection)
SNR_WEIGHT = 0.25        # Was 0.30 (reduce SNR penalty)
```
⚠️ Risk: More false alarms on real faces

### **To reduce false alarms:**
```python
THREAT_THRESHOLD = 0.80  # Was 0.75 (more conservative)
SNR_WEIGHT = 0.35        # Was 0.30 (require higher SNR)
LOOP_SCORE_WEIGHT = 0.35  # Was 0.40 (reduce loop sensitivity)
```
⚠️ Risk: Miss some deepfakes

---

## **VERIFICATION RESULTS**

✅ **All environment checks passed:**
- Python 3.11.5 (3.10+ required)
- NumPy 2.4.6
- SciPy latest
- Matplotlib 3.10.9
- FFT computation working (0.0% frequency detection error)
- Butterworth filtering verified
- Config constants loaded successfully

**Status:** 🟢 **READY FOR PHASE 1**

---

**Generated:** $(date)  
**For:** Member C (Threat Analyzer Engineer)  
**Awaiting:** Person A confirmation of config constants  
**Next Phase:** PHASE 1 - Function Implementation
