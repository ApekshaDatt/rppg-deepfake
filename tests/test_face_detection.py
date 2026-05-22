"""
test_face_detection.py — Person A Test Suite

Tests for:
    - modules/face_detection/face_detector.py
    - modules/face_detection/roi_extractor.py
    - modules/face_detection/__init__.py (DataPacket schema)

Run with:
    python3 -m pytest tests/test_face_detection.py -v

All tests are standalone — no webcam or real video file required.
Uses synthetic numpy images for deterministic, fast execution.
"""

import os
import sys
import time
import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2

from modules.face_detection.face_detector import detect_face, FaceDetector
from modules.face_detection.roi_extractor import extract_roi, create_empty_roi
from modules.face_detection import create_data_packet

# ============================================================================
# Helpers: synthetic test frames
# ============================================================================

def _make_blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank (all-zero) BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_uniform_frame(color: tuple = (120, 120, 120),
                         h: int = 480, w: int = 640) -> np.ndarray:
    """Return a solid-color BGR frame."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _make_face_frame() -> np.ndarray:
    """
    Load a real face image for detection tests using OpenCV's sample data,
    OR fall back to a synthetic white oval on gray if not available.
    """
    # Try to load from cv2 sample images (usually available)
    sample_paths = []
    try:
        sample_paths.append(cv2.samples.findFile("lena.jpg"))
    except cv2.error:
        pass
    try:
        sample_paths.append(cv2.samples.findFile("lena.png"))
    except cv2.error:
        pass
    for path in sample_paths:
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.resize(img, (640, 480))

    # Fallback: draw a synthetic oval to mimic a face
    frame = _make_uniform_frame((80, 80, 80))
    center_x, center_y = 320, 240
    # Draw skin-tone oval (face-like shape)
    cv2.ellipse(frame, (center_x, center_y), (100, 130), 0, 0, 360,
                (150, 190, 220), -1)   # skin tone in BGR
    # Eyes (dark circles)
    cv2.circle(frame, (center_x - 35, center_y - 30), 12, (30, 30, 30), -1)
    cv2.circle(frame, (center_x + 35, center_y - 30), 12, (30, 30, 30), -1)
    # Nose
    cv2.circle(frame, (center_x, center_y + 20), 8, (100, 130, 160), -1)
    return frame


# ============================================================================
# Tests: detect_face()
# ============================================================================

class TestDetectFace:
    """Tests for the stateless detect_face() function."""

    def test_blank_frame_no_face(self):
        """Blank frame should return face_detected=False."""
        frame = _make_blank_frame()
        face_detected, bbox = detect_face(frame)
        assert face_detected is False
        assert bbox == (0, 0, 0, 0)

    def test_uniform_frame_no_face(self):
        """Solid-color frame should return face_detected=False."""
        frame = _make_uniform_frame((128, 64, 32))
        face_detected, bbox = detect_face(frame)
        assert face_detected is False
        assert bbox == (0, 0, 0, 0)

    def test_none_frame_no_crash(self):
        """None frame should not crash — returns False gracefully."""
        frame = np.array([])  # Effectively empty
        face_detected, bbox = detect_face(frame)
        assert face_detected is False
        assert bbox == (0, 0, 0, 0)

    def test_returns_tuple_bbox(self):
        """Return type must be (bool, tuple)."""
        frame = _make_blank_frame()
        face_detected, bbox = detect_face(frame)
        assert isinstance(face_detected, (bool, np.bool_))
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4

    def test_bbox_values_non_negative(self):
        """Bounding box values must be non-negative integers."""
        frame = _make_blank_frame()
        _, bbox = detect_face(frame)
        x, y, w, h = bbox
        assert x >= 0 and y >= 0 and w >= 0 and h >= 0

    def test_returns_largest_face(self):
        """When multiple faces are present, we expect only the largest is returned."""
        # We can't easily force multiple detections without a real image,
        # but we verify the function doesn't crash on a complex frame.
        frame = _make_face_frame()
        face_detected, bbox = detect_face(frame)
        # Result is valid — either True or False depending on image
        assert isinstance(face_detected, (bool, np.bool_))
        assert len(bbox) == 4

    def test_grayscale_input_handled(self):
        """2D (grayscale) frame should be handled without crash."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        face_detected, bbox = detect_face(gray)
        assert face_detected is False
        assert bbox == (0, 0, 0, 0)

    def test_small_frame_no_crash(self):
        """Very small frame should not crash — returns no face."""
        small = np.zeros((10, 10, 3), dtype=np.uint8)
        face_detected, bbox = detect_face(small)
        assert face_detected is False

    def test_large_frame_no_crash(self):
        """Large frame (e.g., 4K) should not crash."""
        big = np.zeros((2160, 3840, 3), dtype=np.uint8)
        face_detected, bbox = detect_face(big)
        assert isinstance(face_detected, (bool, np.bool_))


# ============================================================================
# Tests: FaceDetector class (stateful)
# ============================================================================

class TestFaceDetector:
    """Tests for the stateful FaceDetector class."""

    def test_instantiates_without_error(self):
        """FaceDetector should instantiate cleanly."""
        detector = FaceDetector()
        assert detector is not None

    def test_detect_blank_frame(self):
        """Blank frame should return face_detected=False, bbox=(0,0,0,0)."""
        detector = FaceDetector()
        frame = _make_blank_frame()
        face_detected, bbox = detector.detect(frame)
        assert face_detected is False

    def test_fallback_used_when_face_lost(self):
        """After losing face, fallback bbox should be returned for max_fallback_frames."""
        detector = FaceDetector(max_fallback_frames=3)
        # Manually set _last_bbox to simulate a prior detection
        detector._last_bbox = (100, 100, 150, 150)
        detector._fallback_count = 0

        blank = _make_blank_frame()
        face_detected, bbox = detector.detect(blank)

        # Should use fallback bbox
        assert face_detected is False
        assert bbox == (100, 100, 150, 150)

    def test_fallback_expires_after_limit(self):
        """After max_fallback_frames, fallback should stop being used."""
        detector = FaceDetector(max_fallback_frames=2)
        detector._last_bbox = (100, 100, 150, 150)
        detector._fallback_count = 2  # Already at limit

        blank = _make_blank_frame()
        face_detected, bbox = detector.detect(blank)

        # Fallback count exceeded → return (0,0,0,0)
        assert bbox == (0, 0, 0, 0)

    def test_reset_clears_state(self):
        """Reset should clear last_bbox and fallback_count."""
        detector = FaceDetector()
        detector._last_bbox = (10, 20, 100, 100)
        detector._fallback_count = 3
        detector.reset()

        assert detector._last_bbox == (0, 0, 0, 0)
        assert detector._fallback_count == 0

    def test_detect_returns_correct_types(self):
        """detect() must return (bool, tuple)."""
        detector = FaceDetector()
        frame = _make_blank_frame()
        result = detector.detect(frame)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], tuple)


# ============================================================================
# Tests: extract_roi()
# ============================================================================

class TestExtractROI:
    """Tests for the extract_roi() function."""

    def test_zero_bbox_returns_empty(self):
        """Zero bounding box should return empty ROIs."""
        frame = _make_uniform_frame()
        roi_fore, roi_cheeks = extract_roi(frame, (0, 0, 0, 0))
        assert roi_fore.size == 0
        assert roi_cheeks.size == 0

    def test_invalid_bbox_returns_empty(self):
        """Negative w/h bbox should return empty ROIs."""
        frame = _make_uniform_frame()
        roi_fore, roi_cheeks = extract_roi(frame, (0, 0, -1, -1))
        assert roi_fore.size == 0

    def test_valid_bbox_forehead_shape(self):
        """Forehead ROI should be non-empty with valid face bbox."""
        frame = _make_uniform_frame(h=480, w=640)
        face_bbox = (160, 80, 320, 400)  # 320x400px face centered on frame
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox, padding=0)
        assert roi_fore.size > 0
        assert roi_fore.ndim == 3
        assert roi_fore.shape[2] == 3

    def test_valid_bbox_cheeks_shape(self):
        """Cheeks ROI should be non-empty with valid face bbox."""
        frame = _make_uniform_frame(h=480, w=640)
        face_bbox = (160, 80, 320, 400)
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox, padding=0)
        assert roi_cheeks.size > 0
        assert roi_cheeks.ndim == 3
        assert roi_cheeks.shape[2] == 3

    def test_forehead_height_within_face(self):
        """Forehead ROI height should be ≤ 30% of face height."""
        frame = _make_uniform_frame(h=480, w=640)
        face_h = 300
        face_bbox = (100, 60, 240, face_h)
        roi_fore, _ = extract_roi(frame, face_bbox, padding=0)
        # Forehead height ≤ 30% of face_h
        expected_max_height = int(face_h * 0.30) + 2  # +2 tolerance for rounding
        if roi_fore.size > 0:
            assert roi_fore.shape[0] <= expected_max_height

    def test_empty_frame_returns_empty(self):
        """Empty/None frame should return empty ROIs without crashing."""
        empty = np.array([])
        roi_fore, roi_cheeks = extract_roi(empty, (0, 0, 100, 100))
        assert roi_fore.size == 0
        assert roi_cheeks.size == 0

    def test_large_padding_does_not_crash(self):
        """Padding larger than ROI dimensions should not crash."""
        frame = _make_uniform_frame()
        face_bbox = (100, 100, 200, 200)
        # padding=200 is larger than any sub-region
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox, padding=200)
        # Should return empty arrays (clamped to empty)
        # No crash is the critical requirement here.

    def test_roi_dtype_is_uint8(self):
        """ROI arrays should be uint8 (same as input frame)."""
        frame = _make_uniform_frame()
        face_bbox = (100, 100, 200, 200)
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox, padding=2)
        if roi_fore.size > 0:
            assert roi_fore.dtype == np.uint8
        if roi_cheeks.size > 0:
            assert roi_cheeks.dtype == np.uint8

    def test_roi_values_come_from_frame(self):
        """ROI pixel values should match the source frame pixels."""
        # Fill frame with known color
        known_color = (42, 99, 200)
        frame = _make_uniform_frame(known_color)
        face_bbox = (100, 60, 200, 250)
        roi_fore, _ = extract_roi(frame, face_bbox, padding=0)
        if roi_fore.size > 0:
            # All pixels should be the same known color
            assert np.all(roi_fore[:, :, 0] == known_color[0])
            assert np.all(roi_fore[:, :, 1] == known_color[1])
            assert np.all(roi_fore[:, :, 2] == known_color[2])


# ============================================================================
# Tests: create_empty_roi()
# ============================================================================

class TestCreateEmptyROI:
    """Tests for the create_empty_roi() helper."""

    def test_returns_zero_size_array(self):
        """Should return (0, 0, 3) array."""
        roi = create_empty_roi()
        assert roi.size == 0
        assert roi.ndim == 3
        assert roi.shape == (0, 0, 3)

    def test_dtype_is_uint8(self):
        """Empty ROI should be uint8."""
        roi = create_empty_roi()
        assert roi.dtype == np.uint8


# ============================================================================
# Tests: DataPacket schema (create_data_packet)
# ============================================================================

class TestDataPacket:
    """Tests for the DataPacket interface contract."""

    def _make_packet(self) -> dict:
        frame = _make_blank_frame()
        roi = np.zeros((10, 20, 3), dtype=np.uint8)
        return create_data_packet(
            frame=frame,
            face_detected=True,
            face_bbox=(100, 100, 200, 200),
            roi_forehead=roi,
            roi_cheeks=roi,
        )

    def test_packet_has_all_required_keys(self):
        """DataPacket must have all 6 required keys."""
        packet = self._make_packet()
        required_keys = {"frame", "face_detected", "face_bbox",
                         "roi_forehead", "roi_cheeks", "timestamp"}
        assert required_keys.issubset(set(packet.keys()))

    def test_packet_frame_is_ndarray(self):
        """frame must be np.ndarray."""
        packet = self._make_packet()
        assert isinstance(packet["frame"], np.ndarray)

    def test_packet_face_detected_is_bool(self):
        """face_detected must be bool."""
        packet = self._make_packet()
        assert isinstance(packet["face_detected"], bool)

    def test_packet_face_bbox_is_tuple_of_4(self):
        """face_bbox must be a tuple of 4 elements."""
        packet = self._make_packet()
        assert isinstance(packet["face_bbox"], tuple)
        assert len(packet["face_bbox"]) == 4

    def test_packet_roi_forehead_is_ndarray(self):
        """roi_forehead must be np.ndarray with 3 channels."""
        packet = self._make_packet()
        assert isinstance(packet["roi_forehead"], np.ndarray)

    def test_packet_roi_cheeks_is_ndarray(self):
        """roi_cheeks must be np.ndarray with 3 channels."""
        packet = self._make_packet()
        assert isinstance(packet["roi_cheeks"], np.ndarray)

    def test_packet_timestamp_is_float(self):
        """timestamp must be a float (unix epoch)."""
        packet = self._make_packet()
        assert isinstance(packet["timestamp"], float)

    def test_packet_timestamp_is_recent(self):
        """timestamp should be within 5 seconds of now."""
        packet = self._make_packet()
        delta = abs(time.time() - packet["timestamp"])
        assert delta < 5.0, f"Timestamp too old: {delta:.2f}s"

    def test_packet_custom_timestamp(self):
        """Custom timestamp should be preserved."""
        custom_ts = 1234567890.0
        frame = _make_blank_frame()
        roi = create_empty_roi()
        packet = create_data_packet(
            frame=frame,
            face_detected=False,
            face_bbox=(0, 0, 0, 0),
            roi_forehead=roi,
            roi_cheeks=roi,
            timestamp=custom_ts,
        )
        assert packet["timestamp"] == custom_ts


# ============================================================================
# Integration test: full pipeline stub
# ============================================================================

class TestFullPipelineStub:
    """Integration test: face_detector → roi_extractor → data_packet."""

    def test_pipeline_on_blank_frame(self):
        """Full pipeline on blank frame: no face, safe defaults, valid packet."""
        detector = FaceDetector()
        frame = _make_blank_frame()

        face_detected, face_bbox = detector.detect(frame)
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox)
        packet = create_data_packet(frame, face_detected, face_bbox,
                                    roi_fore, roi_cheeks)

        assert packet["face_detected"] is False
        assert packet["face_bbox"] == (0, 0, 0, 0)
        assert packet["roi_forehead"].size == 0
        assert packet["roi_cheeks"].size == 0

    def test_pipeline_with_manual_bbox(self):
        """Full pipeline with a manually set bbox: ROIs should be non-empty."""
        frame = _make_uniform_frame((100, 150, 200))
        # Manually bypass detection and use a known bbox
        face_bbox = (100, 80, 220, 280)
        roi_fore, roi_cheeks = extract_roi(frame, face_bbox, padding=2)
        packet = create_data_packet(frame, True, face_bbox, roi_fore, roi_cheeks)

        assert packet["face_detected"] is True
        assert packet["face_bbox"] == face_bbox
        assert packet["roi_forehead"].size > 0
        assert packet["roi_cheeks"].size > 0
        assert packet["roi_forehead"].shape[2] == 3
        assert packet["roi_cheeks"].shape[2] == 3


# ============================================================================
# Config sanity checks
# ============================================================================

class TestConfigConstants:
    """Verify Person A config constants are present and valid."""

    def test_roi_padding_is_positive(self):
        from config import ROI_PADDING
        assert ROI_PADDING >= 0

    def test_min_face_size_is_tuple(self):
        from config import MIN_FACE_SIZE
        assert isinstance(MIN_FACE_SIZE, tuple)
        assert len(MIN_FACE_SIZE) == 2

    def test_haar_scale_factor_valid(self):
        from config import HAAR_SCALE_FACTOR
        assert 1.01 <= HAAR_SCALE_FACTOR <= 2.0

    def test_haar_min_neighbors_valid(self):
        from config import HAAR_MIN_NEIGHBORS
        assert 1 <= HAAR_MIN_NEIGHBORS <= 20

    def test_forehead_height_ratio_valid(self):
        from config import ROI_FOREHEAD_HEIGHT_RATIO
        assert 0.1 <= ROI_FOREHEAD_HEIGHT_RATIO <= 0.5

    def test_forehead_width_ratio_valid(self):
        from config import ROI_FOREHEAD_WIDTH_RATIO
        assert 0.3 <= ROI_FOREHEAD_WIDTH_RATIO <= 1.0

    def test_cheek_height_ratio_valid(self):
        from config import ROI_CHEEK_HEIGHT_RATIO
        assert 0.1 <= ROI_CHEEK_HEIGHT_RATIO <= 0.7


if __name__ == "__main__":
    # Run all tests when executed directly
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
