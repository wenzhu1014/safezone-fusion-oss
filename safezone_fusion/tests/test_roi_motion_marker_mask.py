"""Lock in: roi_motion masks out the cart marker before computing motion.

This protects the design decision documented in NOTES.md (2026-06-18 11:10):
single NIR cam can't tell marker from limb, so we explicitly hand the marker
bbox down via packet.meta and have roi_motion zero out that region. If anyone
later changes _select_image / VisionSensorFusion / roi_motion in a way that
breaks the mask path, these tests fail loudly.

Tests use synthetic numpy frames; no real camera.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.interfaces import CameraPacket  # noqa: E402
from src.vision.roi_motion_detector import (  # noqa: E402
    RoiMotionConfig,
    RoiMotionObjectDetector,
)

def _gray_frame(value: int = 60, h: int = 100, w: int = 100) -> np.ndarray:
    """Return a uniform 3-channel BGR frame at the given gray level."""
    return np.full((h, w, 3), value, dtype=np.uint8)

def _make_detector(roi=(0.0, 0.0, 1.0, 1.0), mask_pad: int = 10) -> RoiMotionObjectDetector:
    """Make a detector that triggers fast (warmup=1, min_consecutive=1)."""
    return RoiMotionObjectDetector(
        RoiMotionConfig(
            enabled=True,
            roi=roi,
            min_area_px=50.0,
            threshold=10,
            blur_kernel=1,
            warmup_frames=1,
            min_consecutive=1,
            marker_mask_pad_px=mask_pad,
        )
    )

class RoiMotionMarkerMaskTests(unittest.TestCase):
    def test_motion_outside_marker_still_triggers(self):
        """Hand-style motion outside the marker bbox must trigger STOP."""
        det = _make_detector(mask_pad=5)
        # Frame 1: warmup baseline (uniform background)
        det.detect(CameraPacket(rgb=_gray_frame(60), meta={}))
        # Frame 2: a bright square at top-left (the "hand"), no marker bbox
        f2 = _gray_frame(60)
        f2[10:40, 10:40] = 200  # 30x30 = 900 pixels of motion
        objs = det.detect(CameraPacket(rgb=f2, meta={}))
        self.assertGreater(len(objs), 0, "motion outside marker should trigger object")
        self.assertGreater(det.last_motion_area, 0)

    def test_marker_only_motion_is_masked(self):
        """If the only thing moving is the marker itself, no STOP."""
        det = _make_detector(mask_pad=5)
        det.detect(CameraPacket(rgb=_gray_frame(60), meta={}))
        # Frame 2: bright square ONLY where the marker sits.
        f2 = _gray_frame(60)
        f2[10:40, 10:40] = 200
        # Tell roi_motion that the marker is exactly there.
        meta = {"marker_bbox_px": (10, 10, 40, 40)}
        objs = det.detect(CameraPacket(rgb=f2, meta=meta))
        self.assertEqual(objs, [], "marker-only motion must be masked, no object emitted")
        self.assertEqual(det.last_motion_area, 0.0)
        self.assertIsNotNone(det.last_marker_mask_px, "mask box should be recorded for visualization")

    def test_motion_with_marker_present_but_hand_elsewhere(self):
        """Marker present + hand in a different region: STOP must fire on the hand."""
        det = _make_detector(mask_pad=5)
        det.detect(CameraPacket(rgb=_gray_frame(60), meta={}))
        f2 = _gray_frame(60)
        # Marker region (top-left) lights up
        f2[10:40, 10:40] = 200
        # Hand region (bottom-right) also lights up
        f2[60:90, 60:90] = 200
        meta = {"marker_bbox_px": (10, 10, 40, 40)}
        objs = det.detect(CameraPacket(rgb=f2, meta=meta))
        self.assertGreater(len(objs), 0, "hand motion outside marker mask should still STOP")

    def test_disabled_mask_pad_still_runs(self):
        """marker_mask_pad_px < 0 disables masking; marker movement triggers STOP again (legacy behavior)."""
        det = _make_detector(mask_pad=-1)
        det.detect(CameraPacket(rgb=_gray_frame(60), meta={}))
        f2 = _gray_frame(60)
        f2[10:40, 10:40] = 200
        meta = {"marker_bbox_px": (10, 10, 40, 40)}
        objs = det.detect(CameraPacket(rgb=f2, meta=meta))
        self.assertGreater(len(objs), 0, "with masking disabled, marker movement counts as motion (legacy)")

    def test_no_marker_meta_means_no_masking(self):
        """If meta has no marker_bbox_px, behavior is the same as before this feature."""
        det = _make_detector(mask_pad=5)
        det.detect(CameraPacket(rgb=_gray_frame(60), meta={}))
        f2 = _gray_frame(60)
        f2[10:40, 10:40] = 200
        objs = det.detect(CameraPacket(rgb=f2, meta=None))
        self.assertGreater(len(objs), 0)
        self.assertIsNone(det.last_marker_mask_px)

class VisionSensorFusionMarkerBboxTests(unittest.TestCase):
    """The fusion layer must publish marker_bbox_px into packet.meta."""

    def test_fusion_writes_marker_bbox_into_meta(self):
        # Lazy import; bring along the protocol-y stub objects.
        from src.vision.interfaces import VisionSensorFusion, PirPacket

        class _StubMarkerDetector:
            # numpy corners: shape (1, 4, 2) — one marker, 4 corner points (x,y)
            last_corners = np.array([[[20, 30], [60, 30], [60, 90], [20, 90]]], dtype=np.float32)
            def detect(self, packet):
                return None  # we don't care about MarkerObservation here

        captured = {}

        class _CapturingObjectDetector:
            def detect(self, packet):
                captured["meta"] = dict(packet.meta) if packet.meta else None
                return []

        class _StubPirReader:
            def read(self):
                return PirPacket(triggered=False, timestamp=0.0)

        fusion = VisionSensorFusion(
            marker_detector=_StubMarkerDetector(),
            object_detector=_CapturingObjectDetector(),
            pir_reader=_StubPirReader(),
        )
        fusion.read(CameraPacket(rgb=_gray_frame(60)))
        self.assertIn("marker_bbox_px", captured["meta"] or {})
        self.assertEqual(captured["meta"]["marker_bbox_px"], (20, 30, 60, 90))

if __name__ == "__main__":
    unittest.main()
