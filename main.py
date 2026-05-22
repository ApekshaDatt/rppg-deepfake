"""
main.py — rPPG Deepfake Detection System
Person A: System Architect & Integration Owner

8-Step pipeline loop (per playbook Phase 3 specification):
    1. Capture frame from webcam or video file
    2. Detect face (with 1-frame fallback)
    3. Extract ROI (forehead + cheeks)
    4. Build DataPacket and accumulate frame buffer
    5. Extract rPPG signal (Person B module — stub-safe)
    6. Analyze threat (Person C module — already implemented)
    7. Display overlay (Person D module — stub-safe)
    8. Log result to CSV (Person D module — stub-safe)

Controls:
    Q — quit
    R — reset face tracker and verdict history

Usage:
    # Live webcam (default):
    python3 main.py

    # Video file:
    python3 main.py --source data/test_real/your_face.mp4

    # Specific webcam index:
    python3 main.py --camera 1
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np

# ============================================================================
# Path setup — ensure project root is on sys.path
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Configuration
# ============================================================================
from config import FPS, BUFFER_SIZE, CALIBRATION_FRAMES, VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN

# ============================================================================
# Person A modules (face detection)
# ============================================================================
from modules.face_detection import (
    FaceDetector,
    extract_roi,
    create_data_packet,
)
from modules.face_detection.roi_extractor import draw_roi_overlay

# ============================================================================
# Person C module (threat analyzer) — ALREADY IMPLEMENTED
# ============================================================================
from modules.threat_analyzer import analyze_threat, reset_verdict_history

# ============================================================================
# Stub-safe imports for Person B and Person D modules
# These return safe defaults until those modules are implemented.
# ============================================================================

def _import_rppg_engine():
    """Import rPPG Engine (Person B). Returns stub if module is not ready."""
    try:
        from modules.rppg_engine import process_signal
        return process_signal
    except (ImportError, AttributeError):
        return _rppg_engine_stub

def _import_ui_display():
    """Import UI display (Person D). Returns stub if module is not ready."""
    try:
        from modules.ui.display import draw_verdict_overlay
        return draw_verdict_overlay
    except (ImportError, AttributeError):
        return _display_stub

def _import_ui_logger():
    """Import UI logger (Person D). Returns stub if module is not ready."""
    try:
        from modules.ui.logger import log_result
        return log_result
    except (ImportError, AttributeError):
        return _logger_stub

def _import_ui_plotter():
    """Import UI plotter (Person D). Returns stub if module is not ready."""
    try:
        from modules.ui.plotter import get_signal_overlay
        return get_signal_overlay
    except (ImportError, AttributeError):
        return _plotter_stub


# ============================================================================
# STUBS — safe fallbacks while Person B and D implement their modules
# ============================================================================

def _rppg_engine_stub(roi_forehead: np.ndarray, roi_cheeks: np.ndarray,
                       frame_count: int, buffer_size: int) -> dict:
    """Stub for Person B's rPPG Engine. Returns zeroed-out output dict."""
    is_calibrating = frame_count < CALIBRATION_FRAMES
    return {
        "rppg_signal": np.zeros(buffer_size, dtype=np.float64),
        "estimated_bpm": 0.0,
        "is_calibrating": is_calibrating,
        "signal_quality": 0.0,
        "method_used": "STUB",
    }


def _display_stub(frame: np.ndarray, verdict_dict: dict, face_bbox: tuple,
                  face_detected: bool, frame_count: int) -> np.ndarray:
    """Minimal built-in overlay while Person D implements display.py."""
    overlay = frame.copy()
    x, y, w, h = face_bbox

    verdict = verdict_dict.get("verdict", VERDICT_UNCERTAIN)
    bpm = verdict_dict.get("bpm", 0.0)
    confidence = verdict_dict.get("confidence", 0.0)
    is_calibrating = verdict_dict.get("is_calibrating", False)

    # Color-code bounding box
    if is_calibrating:
        color = (0, 255, 255)   # Yellow — calibrating
        label = "CALIBRATING..."
    elif verdict == VERDICT_REAL:
        color = (0, 220, 0)     # Green — real
        label = f"REAL  {confidence*100:.0f}%"
    elif verdict == VERDICT_THREAT:
        color = (0, 0, 220)     # Red — threat
        label = f"THREAT  {confidence*100:.0f}%"
    else:
        color = (0, 200, 255)   # Amber — uncertain
        label = f"UNCERTAIN  {confidence*100:.0f}%"

    if w > 0 and h > 0:
        thickness = 3 if verdict == VERDICT_THREAT else 2
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

    # Verdict text
    cv2.putText(overlay, label, (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    # BPM text
    if bpm > 0:
        bpm_text = f"BPM: {bpm:.1f}"
        cv2.putText(overlay, bpm_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Method tag (bottom right)
    method = verdict_dict.get("method_used", "")
    if method and method != "STUB":
        h_frame = overlay.shape[0]
        cv2.putText(overlay, f"Method: {method}", (10, h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return overlay


def _logger_stub(timestamp: float, verdict: str, confidence: float,
                 bpm: float, loop_detected: bool) -> None:
    """No-op logger stub while Person D implements logger.py."""
    pass


def _plotter_stub(signal: np.ndarray, frame_width: int) -> np.ndarray | None:
    """No-op plotter stub while Person D implements plotter.py."""
    return None


# ============================================================================
# FPS Counter
# ============================================================================

class FPSCounter:
    """Simple rolling FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        """Record a frame and return current FPS."""
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(source: int | str = 0) -> None:
    """Run the full rPPG deepfake detection pipeline.

    Args:
        source: Camera index (int) or path to video file (str).
    """
    print("=" * 60)
    print("  rPPG DEEPFAKE DETECTION SYSTEM")
    print("  Person A — Integration Owner")
    print("=" * 60)
    print(f"  Source: {'Webcam' if isinstance(source, int) else source}")
    print(f"  FPS target: {FPS}")
    print(f"  Buffer size: {BUFFER_SIZE} samples ({BUFFER_SIZE // FPS}s)")
    print(f"  Calibration: {CALIBRATION_FRAMES} frames ({CALIBRATION_FRAMES // FPS}s)")
    print("  Press Q to quit, R to reset")
    print("=" * 60)

    # Load modules (stub-safe)
    process_signal = _import_rppg_engine()
    draw_verdict_overlay = _import_ui_display()
    log_result = _import_ui_logger()
    get_signal_overlay = _import_ui_plotter()

    using_b_stub = (process_signal is _rppg_engine_stub)
    using_d_stub = (draw_verdict_overlay is _display_stub)
    print(f"  rPPG Engine (B): {'STUB — zeroed signal' if using_b_stub else 'LOADED'}")
    print(f"  UI Display  (D): {'STUB — built-in overlay' if using_d_stub else 'LOADED'}")
    print()

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        print("  For webcam: ensure camera is connected and not in use.")
        print("  For file: check the path is correct.")
        sys.exit(1)

    # Set camera properties if live webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FPS, FPS)

    # Initialize components
    face_detector = FaceDetector()
    fps_counter = FPSCounter()
    frame_count = 0
    last_verdict_dict = {
        "verdict": VERDICT_UNCERTAIN,
        "confidence": 0.0,
        "bpm": 0.0,
        "pulse_present": False,
        "loop_detected": False,
        "snr_score": 0.0,
        "dominant_freq_hz": 0.0,
        "is_calibrating": True,
        "method_used": "",
    }
    last_face_bbox = (0, 0, 0, 0)
    last_face_detected = False

    print("Pipeline running. Warming up...")

    try:
        while True:
            # ================================================================
            # STEP 1: Capture frame
            # ================================================================
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video source. Exiting.")
                break

            timestamp = time.time()
            frame_count += 1
            current_fps = fps_counter.tick()

            # ================================================================
            # STEP 2: Detect face (with tracking fallback)
            # ================================================================
            face_detected, face_bbox = face_detector.detect(frame)
            if face_bbox != (0, 0, 0, 0):
                last_face_bbox = face_bbox
                last_face_detected = face_detected

            # ================================================================
            # STEP 3: Extract ROI
            # ================================================================
            roi_forehead, roi_cheeks = extract_roi(frame, face_bbox)

            # ================================================================
            # STEP 4: Build DataPacket
            # ================================================================
            packet = create_data_packet(
                frame=frame,
                face_detected=face_detected,
                face_bbox=face_bbox,
                roi_forehead=roi_forehead,
                roi_cheeks=roi_cheeks,
                timestamp=timestamp,
            )

            # ================================================================
            # STEP 5: rPPG signal extraction (Person B)
            # ================================================================
            rppg_output = process_signal(
                roi_forehead=packet["roi_forehead"],
                roi_cheeks=packet["roi_cheeks"],
                frame_count=frame_count,
                buffer_size=BUFFER_SIZE,
            )

            rppg_signal: np.ndarray = rppg_output.get(
                "rppg_signal", np.zeros(BUFFER_SIZE, dtype=np.float64)
            )
            estimated_bpm: float = rppg_output.get("estimated_bpm", 0.0)
            is_calibrating: bool = rppg_output.get("is_calibrating", frame_count < CALIBRATION_FRAMES)
            signal_quality: float = rppg_output.get("signal_quality", 0.0)
            method_used: str = rppg_output.get("method_used", "")

            # ================================================================
            # STEP 6: Threat analysis (Person C — fully implemented)
            # ================================================================
            verdict_dict = analyze_threat(
                signal=rppg_signal,
                estimated_bpm=estimated_bpm,
                is_calibrating=is_calibrating,
                signal_quality=signal_quality,
                fs=FPS,
            )
            verdict_dict["is_calibrating"] = is_calibrating
            verdict_dict["method_used"] = method_used
            last_verdict_dict = verdict_dict

            # ================================================================
            # STEP 7: Display overlay (Person D or built-in stub)
            # ================================================================
            display_frame = draw_verdict_overlay(
                frame=frame,
                verdict_dict=verdict_dict,
                face_bbox=face_bbox,
                face_detected=face_detected,
                frame_count=frame_count,
            )

            # Draw ROI region indicators (Person A debug overlay)
            if face_bbox != (0, 0, 0, 0):
                display_frame = draw_roi_overlay(
                    display_frame, face_bbox, face_detected
                )

            # FPS indicator (top right)
            h_f, w_f = display_frame.shape[:2]
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(display_frame, fps_text, (w_f - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Module status indicators (top right, below FPS)
            status_color = (100, 255, 100) if not using_b_stub else (100, 100, 255)
            status_text = "B:LIVE" if not using_b_stub else "B:STUB"
            cv2.putText(display_frame, status_text, (w_f - 90, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

            # Optionally overlay rPPG signal graph from Person D
            signal_overlay = get_signal_overlay(rppg_signal, w_f)
            if signal_overlay is not None and signal_overlay.size > 0:
                oh = signal_overlay.shape[0]
                display_frame[h_f - oh:h_f, :w_f] = signal_overlay

            # Show frame
            cv2.imshow("rPPG Deepfake Detector", display_frame)

            # ================================================================
            # STEP 8: Log result (Person D or no-op stub)
            # ================================================================
            v = verdict_dict.get("verdict", VERDICT_UNCERTAIN)
            b = verdict_dict.get("bpm", 0.0)
            c = verdict_dict.get("confidence", 0.0)
            
            try:
                # New logger.py expects 4 args: (verdict, bpm, confidence, pulse)
                bpm_str = f"{b:.1f} BPM" if b > 0 else "N/A"
                conf_str = f"{c*100:.0f}%"
                pulse_str = "STABLE" if v == VERDICT_REAL else ("UNSTABLE" if v == VERDICT_THREAT else "WEAK")
                
                log_result(v, bpm_str, conf_str, pulse_str)
            except TypeError:
                # Fallback to stub signature
                log_result(
                    timestamp=timestamp,
                    verdict=v,
                    confidence=c,
                    bpm=b,
                    loop_detected=verdict_dict.get("loop_detected", False),
                )

            # ================================================================
            # Key handler
            # ================================================================
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # Q or Esc
                print("\nUser requested exit.")
                break
            elif key == ord("r") or key == ord("R"):
                print("\nResetting tracker and verdict history...")
                face_detector.reset()
                reset_verdict_history()
                frame_count = 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped cleanly.")
        print(f"Processed {frame_count} frames.")
        print(f"Last verdict: {last_verdict_dict.get('verdict')}  "
              f"BPM: {last_verdict_dict.get('bpm')}  "
              f"Confidence: {last_verdict_dict.get('confidence')}")


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="rPPG Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                          # Live webcam (index 0)
  python3 main.py --camera 1              # Second webcam
  python3 main.py --source video.mp4      # Video file
  python3 main.py --source data/test_real/face.mp4
        """
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Path to video file. If not set, uses live webcam."
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Webcam device index (default: 0). Ignored if --source is set."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.source is not None:
        video_source: int | str = args.source
    else:
        video_source = args.camera

    run_pipeline(source=video_source)
