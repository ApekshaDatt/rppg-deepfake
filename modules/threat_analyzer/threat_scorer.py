"""
Threat Scorer: Aggregation layer for rPPG deepfake threat detection.

Takes outputs from ``fft_analyzer`` and ``pattern_detector``, combines them
with a weighted scoring formula, applies a rolling majority vote over a
configurable window, and returns the final verdict dictionary consumed by
Person D (UI) and ``main.py``.

Public API
----------
  score_threat(fft_result, loop_result)  → dict   (verdict, confidence, …)
  reset_verdict_history()                → None   (clear rolling window)
"""

import sys
import os
from collections import deque, Counter

# ---------------------------------------------------------------------------
# Path setup — standalone execution or import from any CWD.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import THREAT_THRESHOLD, SNR_THRESHOLD, VERDICT_WINDOW


# ===========================================================================
# Module-level rolling verdict history
# ===========================================================================
_verdict_history: deque = deque(maxlen=VERDICT_WINDOW)


# ===========================================================================
# Private helpers
# ===========================================================================

def _compute_snr_threat(snr: float) -> float:
    """Return a suspicion score in [0.0, 1.0] based on the SNR value.

    * SNR near zero (flat / dead signal)       → 1.0  (very suspicious)
    * SNR in normal biological range            → 0.0  (healthy)
    * SNR unnaturally high (synthetic purity)   → 1.0  (very suspicious)

    "Unnaturally high" is defined as > SNR_THRESHOLD × 3.
    """
    if snr <= 0.0:
        # Dead / flat signal — maximally suspicious
        return 1.0

    upper_bound = SNR_THRESHOLD * 3.0

    if snr > upper_bound:
        # Unnaturally clean — synthetic purity
        return 1.0

    if snr >= SNR_THRESHOLD:
        # Normal biological range — not suspicious
        return 0.0

    # Below threshold but above zero → linearly suspicious
    # (the lower the SNR, the more suspicious)
    return 1.0 - (snr / SNR_THRESHOLD)


def _majority_vote(history: deque) -> str:
    """Return the most common verdict with conservative tiebreaking.

    Tiebreak order: THREAT > UNCERTAIN > REAL
    (prefer false alarm over missed detection)
    """
    if not history:
        return "UNCERTAIN"

    counts = Counter(history)
    max_count = max(counts.values())

    # Tiebreak precedence
    for verdict in ("THREAT", "UNCERTAIN", "REAL"):
        if counts.get(verdict, 0) == max_count:
            return verdict

    return "UNCERTAIN"  # fallback


# ===========================================================================
# Public API
# ===========================================================================

def score_threat(fft_result: dict, loop_result: dict) -> dict:
    """Aggregate detection signals into a final verdict.

    Parameters
    ----------
    fft_result : dict
        Keys: dominant_freq_hz, snr_score, pulse_present, estimated_bpm
        (output of ``fft_analyzer.analyze_fft``)
    loop_result : dict
        Keys: loop_detected, loop_score
        (output of ``pattern_detector.detect_loop``)

    Returns
    -------
    dict
        verdict          : str    – majority-voted verdict (REAL / THREAT / UNCERTAIN)
        confidence       : float  – proportion of history matching majority [0.0, 1.0]
        bpm              : float  – estimated BPM (0.0 if no pulse)
        pulse_present    : bool
        loop_detected    : bool
        snr_score        : float
        dominant_freq_hz : float
    """

    # Unpack inputs
    pulse_present    = fft_result.get("pulse_present", False)
    snr_score        = fft_result.get("snr_score", 0.0)
    dominant_freq_hz = fft_result.get("dominant_freq_hz", 0.0)
    estimated_bpm    = fft_result.get("estimated_bpm", 0.0)
    loop_detected    = loop_result.get("loop_detected", False)
    loop_score       = loop_result.get("loop_score", 0.0)

    # ── (a) Compute raw threat score ──────────────────────────────────────
    no_pulse_component = 0.40 if not pulse_present else 0.0
    loop_component = 0.40 * float(loop_score) if loop_detected else 0.0
    snr_component = 0.20 * _compute_snr_threat(snr_score)

    # Loop override: a confirmed synthetic loop is the strongest deepfake
    # indicator — a deepfake can synthesise a fake pulse but cannot hide
    # periodic repetition from autocorrelation.  This boost ensures that
    # loop_detected=True can push the score above THREAT_THRESHOLD even
    # when pulse_present=True (max without boost = 0.60 < 0.75).
    loop_override_boost = 0.25 if loop_detected else 0.0

    raw_threat_score = (no_pulse_component + loop_component
                        + snr_component + loop_override_boost)
    raw_threat_score = max(0.0, min(1.0, raw_threat_score))  # clamp [0, 1]

    # ── (b) Map raw score to this frame's verdict ─────────────────────────
    if raw_threat_score >= THREAT_THRESHOLD:
        frame_verdict = "THREAT"
    elif raw_threat_score <= (1.0 - THREAT_THRESHOLD):
        frame_verdict = "REAL"
    else:
        frame_verdict = "UNCERTAIN"

    # ── (c) Append to rolling history ─────────────────────────────────────
    _verdict_history.append(frame_verdict)

    # ── (d) Majority vote over history ────────────────────────────────────
    final_verdict = _majority_vote(_verdict_history)

    # ── (e) Confidence = proportion of history matching majority ──────────
    if len(_verdict_history) > 0:
        match_count = sum(1 for v in _verdict_history if v == final_verdict)
        confidence = match_count / len(_verdict_history)
    else:
        confidence = 0.0

    # ── (f) Return complete output dict ───────────────────────────────────
    return {
        "verdict":          final_verdict,
        "confidence":       round(confidence, 2),
        "bpm":              float(estimated_bpm) if pulse_present else 0.0,
        "pulse_present":    bool(pulse_present),
        "loop_detected":    bool(loop_detected),
        "snr_score":        float(snr_score),
        "dominant_freq_hz": float(dominant_freq_hz),
    }


def reset_verdict_history() -> None:
    """Clear the rolling verdict history.

    Called by ``main.py`` when switching input sources (webcam ↔ file)
    so that stale verdicts from a previous source do not contaminate the
    new source's analysis.
    """
    _verdict_history.clear()


# ===========================================================================
# Standalone self-test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Threat Scorer — self-test suite")
    print("=" * 70)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test Sequence 1 — Real person (10 frames)
    # ------------------------------------------------------------------
    print("\n[Sequence 1] Real person — 10 frames")
    reset_verdict_history()

    fft_real = {
        "dominant_freq_hz": 1.2,
        "snr_score": 8.0,
        "pulse_present": True,
        "estimated_bpm": 72.0,
    }
    loop_real = {"loop_detected": False, "loop_score": 0.3}

    for i in range(10):
        result1 = score_threat(fft_real, loop_real)

    verdict_ok = result1["verdict"] == "REAL"
    test1_pass = verdict_ok
    passed += test1_pass
    failed += (not test1_pass)

    print(f"  verdict    = {result1['verdict']}  (expected REAL)  "
          f"{'✓' if verdict_ok else '✗'}")
    print(f"  confidence = {result1['confidence']:.2f}")
    print(f"  bpm        = {result1['bpm']:.1f}")
    print(f"  → {'PASS' if test1_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test Sequence 2 — Deepfake flat signal (10 frames)
    # ------------------------------------------------------------------
    print("\n[Sequence 2] Deepfake flat signal — 10 frames")
    reset_verdict_history()

    fft_flat = {
        "dominant_freq_hz": 0.0,
        "snr_score": 0.1,
        "pulse_present": False,
        "estimated_bpm": 0.0,
    }
    loop_flat = {"loop_detected": True, "loop_score": 0.97}

    for i in range(10):
        result2 = score_threat(fft_flat, loop_flat)

    verdict_ok = result2["verdict"] == "THREAT"
    test2_pass = verdict_ok
    passed += test2_pass
    failed += (not test2_pass)

    print(f"  verdict    = {result2['verdict']}  (expected THREAT)  "
          f"{'✓' if verdict_ok else '✗'}")
    print(f"  confidence = {result2['confidence']:.2f}")
    print(f"  bpm        = {result2['bpm']:.1f}")
    print(f"  → {'PASS' if test2_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test Sequence 3 — Deepfake perfect loop (10 frames)
    # ------------------------------------------------------------------
    print("\n[Sequence 3] Deepfake perfect loop — 10 frames")
    reset_verdict_history()

    fft_loop = {
        "dominant_freq_hz": 1.0,
        "snr_score": 45.0,
        "pulse_present": True,
        "estimated_bpm": 60.0,
    }
    loop_loop = {"loop_detected": True, "loop_score": 0.98}

    for i in range(10):
        result3 = score_threat(fft_loop, loop_loop)

    verdict_ok = result3["verdict"] == "THREAT"
    test3_pass = verdict_ok
    passed += test3_pass
    failed += (not test3_pass)

    print(f"  verdict    = {result3['verdict']}  (expected THREAT)  "
          f"{'✓' if verdict_ok else '✗'}")
    print(f"  confidence = {result3['confidence']:.2f}")
    print(f"  bpm        = {result3['bpm']:.1f}")
    print(f"  → {'PASS' if test3_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
