"""Lock in: ArUco detector uses packet.nir / packet.mono when RGB is missing.

Why this test exists:
  In dual-camera mode the secondary cable is often a NIR/mono lens. If the
  primary RGB lens is briefly blinded (strong light, lens occluded, glare)
  or simply missing, we still want marker detection to keep working from
  the NIR/mono stream. This test makes sure nobody silently changes that
  fallback order in _select_image.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.aruco_detector import OpenCVArucoMarkerDetector  # noqa: E402
from src.vision.interfaces import CameraPacket  # noqa: E402

def _stub_frame(value: int = 200) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)

class ArucoFallbackOrderTests(unittest.TestCase):
    def setUp(self):
        # Construct with all defaults; we never actually call detect() on a real
        # marker so authorized_ids etc do not matter.
        self.det = OpenCVArucoMarkerDetector()

    def test_rgb_preferred_when_present(self):
        packet = CameraPacket(rgb=_stub_frame(200), nir=_stub_frame(50), mono=_stub_frame(60))
        img = self.det._select_image(packet)
        np.testing.assert_array_equal(img, packet.rgb)

    def test_falls_back_to_left_rgb_then_right_rgb(self):
        packet = CameraPacket(rgb=None, left_rgb=_stub_frame(150), right_rgb=_stub_frame(40))
        img = self.det._select_image(packet)
        np.testing.assert_array_equal(img, packet.left_rgb)

        packet2 = CameraPacket(rgb=None, left_rgb=None, right_rgb=_stub_frame(40))
        img2 = self.det._select_image(packet2)
        np.testing.assert_array_equal(img2, packet2.right_rgb)

    def test_falls_back_to_mono_when_rgb_missing(self):
        packet = CameraPacket(rgb=None, left_rgb=None, right_rgb=None,
                              mono=_stub_frame(60), nir=_stub_frame(40))
        img = self.det._select_image(packet)
        np.testing.assert_array_equal(img, packet.mono)

    def test_falls_back_to_nir_when_only_nir_present(self):
        # This is the dual-camera "secondary cable is NIR" case where the RGB
        # primary just dropped. Detection must still try.
        packet = CameraPacket(rgb=None, left_rgb=None, right_rgb=None,
                              mono=None, nir=_stub_frame(40))
        img = self.det._select_image(packet)
        np.testing.assert_array_equal(img, packet.nir)

    def test_returns_none_when_no_stream(self):
        packet = CameraPacket()
        self.assertIsNone(self.det._select_image(packet))

if __name__ == "__main__":
    unittest.main()
