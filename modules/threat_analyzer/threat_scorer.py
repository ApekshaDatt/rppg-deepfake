"""
Threat Scorer: Fuses multi-modal evidence into threat score and deepfake verdict.
Combines pulse presence, loop correlation, SNR, and signal quality into unified risk metric.
"""

import numpy as np
from config import (
    PULSE_PRESENCE_WEIGHT, LOOP_SCORE_WEIGHT, SNR_WEIGHT, SIGNAL_QUALITY_WEIGHT,
    SNR_THRESHOLD, THREAT_THRESHOLD, THREAT_SCORE_REAL_BOUNDARY,
    BASE_CONFIDENCE, CONF_BONUS_HIGH_SNR, CONF_BONUS_CLEAR_SCORE,
    CONF_BONUS_PULSE_PRESENT, CONF_BONUS_GOOD_QUALITY, SIGNAL_QUALITY_THRESHOLD,
    VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN, MIN_SIGNAL_LENGTH
)


def compute_threat_score(
    pulse_present: bool,
    loop_correlation: float,
    snr_db: float,
    signal_quality: float
) -> float:
    """Compute threat score via weighted fusion of 4 components. Returns [0.0, 1.0]."""
    # Component 1: Pulse presence (0 or 1)
    pulse_component = float(pulse_present) * PULSE_PRESENCE_WEIGHT
    
    # Component 2: Loop correlation (normalized to [0, 1])
    loop_component = np.clip(loop_correlation, 0.0, 1.0) * LOOP_SCORE_WEIGHT
    
    # Component 3: SNR (inverse: higher SNR = lower threat)
    # If SNR >= SNR_THRESHOLD, threat contribution = 0 (good signal)
    # If SNR < SNR_THRESHOLD, threat contribution grows as SNR decreases
    snr_threat_ratio = np.clip((SNR_THRESHOLD - snr_db) / SNR_THRESHOLD, 0.0, 1.0)
    snr_component = snr_threat_ratio * SNR_WEIGHT
    
    # Component 4: Signal quality (inverse: higher quality = lower threat)
    # signal_quality is [0, 1] from rPPG Engine
    signal_quality_component = (1.0 - np.clip(signal_quality, 0.0, 1.0)) * SIGNAL_QUALITY_WEIGHT
    
    # Fuse all components
    threat_score = pulse_component + loop_component + snr_component + signal_quality_component
    
    # Clamp to valid range [0, 1]
    threat_score = np.clip(threat_score, 0.0, 1.0)
    
    return float(threat_score)


def compute_confidence(
    snr_db: float,
    threat_score: float,
    pulse_present: bool,
    signal_quality: float
) -> float:
    """Compute confidence in the verdict. Returns [0.0, 1.0]."""
    confidence = BASE_CONFIDENCE  # Start at 0.5
    
    # Bonus 1: High SNR → more confident
    if snr_db > SNR_THRESHOLD:
        confidence += CONF_BONUS_HIGH_SNR
    
    # Bonus 2: Threat score far from threshold → more confident
    # If threat_score is near THREAT_THRESHOLD (0.75), decision is uncertain
    distance_from_threshold = abs(threat_score - THREAT_THRESHOLD)
    if distance_from_threshold > 0.15:  # > 0.15 away from threshold
        confidence += CONF_BONUS_CLEAR_SCORE
    
    # Bonus 3: Pulse present → more confident
    if pulse_present:
        confidence += CONF_BONUS_PULSE_PRESENT
    
    # Bonus 4: High signal quality → more confident
    if signal_quality > SIGNAL_QUALITY_THRESHOLD:
        confidence += CONF_BONUS_GOOD_QUALITY
    
    # Clamp to valid range [0, 1]
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return float(confidence)


def generate_verdict_and_confidence(
    threat_score: float,
    pulse_present: bool,
    is_calibrating: bool,
    signal_length: int
) -> tuple[str, float]:
    """Map threat_score to verdict category with confidence. Returns (verdict_str, confidence_float)."""
    # Guard 1: Calibration period
    if is_calibrating:
        return (VERDICT_UNCERTAIN, 0.0)
    
    # Guard 2: Insufficient signal length
    if signal_length < MIN_SIGNAL_LENGTH:
        return (VERDICT_UNCERTAIN, 0.0)
    
    # Guard 3: No pulse detected
    if not pulse_present:
        return (VERDICT_UNCERTAIN, 0.3)
    
    # Thresholding logic
    if threat_score >= THREAT_THRESHOLD:
        # High threat → THREAT verdict (red)
        verdict = VERDICT_THREAT
    elif threat_score <= THREAT_SCORE_REAL_BOUNDARY:
        # Low threat → REAL verdict (green)
        verdict = VERDICT_REAL
    else:
        # Middle ground → UNCERTAIN verdict (yellow, safe)
        verdict = VERDICT_UNCERTAIN
    
    # Compute confidence (independent of verdict, reflects overall certainty)
    confidence = compute_confidence(
        snr_db=0.0,  # Note: SNR not passed here; confidence computed in orchestrator
        threat_score=threat_score,
        pulse_present=pulse_present,
        signal_quality=0.5  # Note: signal_quality not passed; confidence computed in orchestrator
    )
    
    return (verdict, confidence)
