"""
UI Module — Person D (Frontend & Presentation Specialist)

This __init__.py defines the expected interface for Person D's UI module.
main.py imports display, logger, and plotter functions from this module.

# =============================================================================
# DISPLAY CONTRACT (draw_verdict_overlay)
# =============================================================================
# Input:
#   frame:         np.ndarray  — (H, W, 3) BGR frame to draw on
#   verdict_dict:  dict        — from threat_analyzer (see threat_analyzer/__init__.py)
#   face_bbox:     tuple       — (x, y, w, h) from face_detector
#   face_detected: bool        — True if detected this frame
#   frame_count:   int         — for calibration display logic
#
# Output:
#   np.ndarray — annotated frame with overlays
#
# Required overlays (Person D checklist):
#   ☐ Colored bounding box (green=REAL, red=THREAT, yellow=UNCERTAIN)
#   ☐ Verdict text in large font (font scale ≥ 1.5) at top of frame
#   ☐ BPM and confidence percentage as text overlay
#   ☐ "CALIBRATING..." during warm-up period
#
# =============================================================================
# LOGGER CONTRACT (log_result)
# =============================================================================
# Input:
#   timestamp:    float   — time.time()
#   verdict:      str     — "REAL", "THREAT", "UNCERTAIN"
#   confidence:   float   — 0.0–1.0
#   bpm:          float   — estimated BPM
#   loop_detected: bool   — True if periodic loop found
#
# Output: None (writes to results.csv)
#
# =============================================================================
# PLOTTER CONTRACT (get_signal_overlay)
# =============================================================================
# Input:
#   signal:      np.ndarray  — rPPG time series to plot
#   frame_width: int          — width of main frame (for sizing)
#
# Output:
#   np.ndarray | None — (h, w, 3) BGR image of graph, or None
# =============================================================================

__all__ = []
"""
