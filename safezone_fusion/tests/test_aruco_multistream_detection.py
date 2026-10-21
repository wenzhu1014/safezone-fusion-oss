"""Dual-camera ArUco fallback must continue past RGB detection failure.

Code-level fallback order tests only prove _select_image ordering. This test
proves the real detector behavior: RGB exists but contains no marker, while
NIR/mono contains a printed-style ArUco marker. The detector must find the NIR
marker and record last_stream_name='nir'.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.aruco_detector import OpenCVArucoMarkerDetector  # noqa: E402
from src.vision.interfaces import CameraPacket  # noqa: E402

def _blank_bgr(value: int = 220, h: int = 720, w: int = 1280) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)

def _frame_with_marker(marker_id: int = 123, h: int = 720, w: int = 1280, size: int = 220) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    frame = np.full((h, w), 255, dtype=np.uint8)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    frame[y0:y0 + size, x0:x0 + size] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

class ArucoMultiStreamDetectionTests(unittest.TestCase):
    def test_rgb_present_but_blocked_falls_through_to_nir(self):
        det = OpenCVArucoMarkerDetector(target_marker_ids={123})
        rgb_blocked = _blank_bgr(30)  # camera exists but sees no marker
        nir_with_marker = _frame_with_marker(123)

        # Hysteresis (2026-06-22): RGB failing for a single frame should NOT
        # immediately jump to NIR (avoids parallax coordinate jitter). It must
        # take _FALLBACK_HOLD_FRAMES consecutive RGB failures before switching.
        obs = None
        for _ in range(det._FALLBACK_HOLD_FRAMES):
            obs = det.detect(CameraPacket(rgb=rgb_blocked, nir=nir_with_marker, timestamp=1.0))

        self.assertIsNotNone(obs)
        self.assertEqual(obs.marker_id, 123)
        self.assertEqual(det.last_stream_name, "nir")
        self.assertIn(123, det.last_detected_ids)

    def test_single_rgb_failure_does_not_immediately_switch_to_nir(self):
        """Hysteresis: one dropped RGB frame is treated as momentary
        no-detection, not an instant switch to the parallax-offset NIR stream."""
        det = OpenCVArucoMarkerDetector(target_marker_ids={123})
        rgb_blocked = _blank_bgr(30)
        nir_with_marker = _frame_with_marker(123)

        # First RGB failure: must NOT switch to NIR yet (hold frames not reached).
        obs = det.detect(CameraPacket(rgb=rgb_blocked, nir=nir_with_marker, timestamp=1.0))
        self.assertIsNone(obs)
        self.assertNotEqual(det.last_stream_name, "nir")

    def test_rgb_recovers_switches_back_from_nir(self):
        """Once RGB sees the marker again, detection returns to RGB immediately
        (RGB is always preferred when present)."""
        det = OpenCVArucoMarkerDetector(target_marker_ids={123})
        rgb_blocked = _blank_bgr(30)
        rgb_with_marker = _frame_with_marker(123)
        nir_with_marker = _frame_with_marker(123)

        # Drive into NIR fallback first.
        for _ in range(det._FALLBACK_HOLD_FRAMES):
            det.detect(CameraPacket(rgb=rgb_blocked, nir=nir_with_marker, timestamp=1.0))
        self.assertEqual(det.last_stream_name, "nir")

        # RGB recovers -> back to rgb on the very next frame.
        obs = det.detect(CameraPacket(rgb=rgb_with_marker, nir=nir_with_marker, timestamp=2.0))
        self.assertIsNotNone(obs)
        self.assertEqual(det.last_stream_name, "rgb")

    def test_rgb_marker_still_preferred_when_available(self):
        det = OpenCVArucoMarkerDetector(target_marker_ids={123})
        rgb_with_marker = _frame_with_marker(123)
        nir_with_marker = _frame_with_marker(123)

        obs = det.detect(CameraPacket(rgb=rgb_with_marker, nir=nir_with_marker, timestamp=1.0))

        self.assertIsNotNone(obs)
        self.assertEqual(obs.marker_id, 123)
        self.assertEqual(det.last_stream_name, "rgb")

if __name__ == "__main__":
    unittest.main()
