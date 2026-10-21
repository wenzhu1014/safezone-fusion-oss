"""Dual-camera capture wiring & skew tests.

These tests do NOT touch real hardware. They use a fake VideoCapture so they
are deterministic and CI-friendly. The point is to lock in:

  1. Single-camera mode still works after the dual-grab refactor.
  2. Dual-camera mode populates rgb + nir/mono slots correctly.
  3. packet.meta carries ts_primary / ts_secondary / skew_ms.
  4. Inter-camera skew can be inspected (downstream logs / alarms can act on it).
  5. A dropped secondary frame downgrades gracefully (no exception, no skew).
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import opencv_capture as cap_mod  # noqa: E402

def _rgb_frame(value: int = 200) -> np.ndarray:
    f = np.zeros((48, 64, 3), dtype=np.uint8)
    # make B != G != R so guess_frame_role -> "rgb"
    f[:, :, 0] = value
    f[:, :, 1] = max(0, value - 80)
    f[:, :, 2] = max(0, value - 160)
    return f

def _mono_like_frame(value: int = 120) -> np.ndarray:
    f = np.full((48, 64, 3), value, dtype=np.uint8)  # B==G==R -> mono_or_nir
    return f

class FakeCapture:
    """Stand-in for cv2.VideoCapture supporting grab/retrieve/read."""

    def __init__(self, frame_factory, *, drop_every: int = 0):
        self._frame_factory = frame_factory
        self._drop_every = drop_every
        self._n = 0
        self._last_frame = None
        self.released = False
        self.props = {}

    # cv2.VideoCapture API surface we use
    def isOpened(self):
        return not self.released

    def set(self, prop_id, value):
        self.props[prop_id] = value
        return True

    def grab(self):
        self._n += 1
        if self._drop_every and self._n % self._drop_every == 0:
            self._last_frame = None
            return False
        self._last_frame = self._frame_factory(self._n)
        return True

    def retrieve(self):
        if self._last_frame is None:
            return False, None
        return True, self._last_frame

    def read(self):
        ok = self.grab()
        if not ok:
            return False, None
        return self.retrieve()

    def release(self):
        self.released = True

class DualCaptureWiringTests(unittest.TestCase):
    def _patch(self, primary_factory, secondary_factory, *, drop_secondary_every: int = 0):
        primary = FakeCapture(primary_factory)
        secondary = FakeCapture(secondary_factory, drop_every=drop_secondary_every) if secondary_factory else None

        def _factory(index, backend):
            if index == 0:
                return primary
            if index == 1 and secondary is not None:
                return secondary
            stub = FakeCapture(lambda _n: _rgb_frame(50))
            stub.released = True  # make isOpened()->False
            return stub

        return patch.object(cap_mod.cv2, "VideoCapture", side_effect=_factory), primary, secondary

    def test_single_camera_still_works(self):
        ctx, primary, _ = self._patch(lambda n: _rgb_frame(200), None)
        with ctx:
            cam = cap_mod.OpenCVCameraCapture(primary_index=0, secondary_index=None, primary_role="auto")
            packet = cam.read()
            cam.release()

        self.assertIsNotNone(packet.rgb)
        self.assertIsNone(packet.nir)
        self.assertEqual(packet.meta["primary_index"], 0)
        self.assertIsNone(packet.meta["secondary_index"])
        self.assertIn("ts_primary", packet.meta)
        self.assertIsNone(packet.meta["ts_secondary"])
        self.assertIsNone(packet.meta["skew_ms"])

    def test_dual_camera_fills_rgb_and_mono(self):
        ctx, primary, secondary = self._patch(
            lambda n: _rgb_frame(200),       # primary -> rgb
            lambda n: _mono_like_frame(120),  # secondary -> mono/nir
        )
        with ctx:
            cam = cap_mod.OpenCVCameraCapture(
                primary_index=0,
                secondary_index=1,
                primary_role="auto",
                secondary_role="auto",
            )
            t0 = time.time()
            packet = cam.read()
            t1 = time.time()
            cam.release()

        self.assertIsNotNone(packet.rgb, "primary RGB should land in packet.rgb")
        self.assertIsNotNone(packet.nir, "secondary mono/NIR should land in packet.nir")
        self.assertIsNotNone(packet.mono, "mono slot should be filled too for compat")
        self.assertEqual(packet.meta["secondary_index"], 1)
        self.assertIsNotNone(packet.meta["ts_primary"])
        self.assertIsNotNone(packet.meta["ts_secondary"])
        self.assertIsNotNone(packet.meta["skew_ms"])
        # ts_primary <= ts_secondary in our grab order; both within this read() window
        self.assertGreaterEqual(packet.meta["ts_secondary"], packet.meta["ts_primary"])
        self.assertGreaterEqual(packet.meta["ts_primary"], t0 - 0.001)
        self.assertLessEqual(packet.meta["ts_secondary"], t1 + 0.001)
        # In a synchronous test we expect very small skew (< 50 ms is generous)
        self.assertLess(abs(packet.meta["skew_ms"]), 50.0)

    def test_dropped_secondary_frame_downgrades_gracefully(self):
        # Secondary fails on first grab (drop_every=1) -> no exception, skew=None
        ctx, primary, secondary = self._patch(
            lambda n: _rgb_frame(200),
            lambda n: _mono_like_frame(120),
            drop_secondary_every=1,
        )
        with ctx:
            cam = cap_mod.OpenCVCameraCapture(primary_index=0, secondary_index=1)
            packet = cam.read()
            cam.release()

        self.assertIsNotNone(packet.rgb, "primary should still produce a frame")
        self.assertIsNone(packet.nir, "secondary dropped -> nir slot empty")
        self.assertIsNone(
            packet.meta["ts_secondary"],
            "dropped secondary should leave ts_secondary=None so monitors can spot it",
        )
        self.assertIsNone(packet.meta["skew_ms"])

    def test_swapped_roles_dual(self):
        # Primary is mono/NIR, secondary is RGB. Should still classify correctly.
        ctx, primary, secondary = self._patch(
            lambda n: _mono_like_frame(120),
            lambda n: _rgb_frame(200),
        )
        with ctx:
            cam = cap_mod.OpenCVCameraCapture(primary_index=0, secondary_index=1)
            packet = cam.read()
            cam.release()
        self.assertIsNotNone(packet.rgb, "rgb slot should pick up the colorful one")
        self.assertIsNotNone(packet.nir, "nir slot should pick up the mono-like one")

if __name__ == "__main__":
    unittest.main()
