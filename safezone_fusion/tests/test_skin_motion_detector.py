from __future__ import annotations

import numpy as np

from safezone_fusion.src.vision.interfaces import CameraPacket
from safezone_fusion.src.vision.skin_motion_detector import SkinMotionObjectDetector

def _make_skin_color_block(shape, color_bgr):
    frame = np.zeros(shape, dtype=np.uint8)
    return frame, np.array(color_bgr, dtype=np.uint8)

def test_skin_motion_disabled_emits_no_objects():
    detector = SkinMotionObjectDetector.from_dict({"enabled": False})
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detector.detect(CameraPacket(rgb=frame, timestamp=0.0)) == []

def test_skin_motion_triggers_on_skin_colored_motion():
    detector = SkinMotionObjectDetector.from_dict(
        {
            "enabled": True,
            "roi": [0.25, 0.25, 0.75, 0.75],
            "min_area_px": 200,
            "motion_threshold": 10,
            "warmup_frames": 1,
            "skin_required": True,
            "skin_overlap_ratio_min": 0.10,
        }
    )
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    detector.detect(CameraPacket(rgb=base.copy(), timestamp=0.0))

    moved = base.copy()
    # BGR skin-ish tone (warm pink), inside ROI region
    moved[40:90, 50:120] = (130, 150, 220)
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.1))
    assert len(objects) == 1
    assert objects[0].label == "motion_skin"  # honest default label since 2026-06-22 semantic-path change
    assert detector.last_triggered is True
    assert detector.last_skin_overlap_ratio > 0.0

def test_skin_motion_rejects_non_skin_motion():
    detector = SkinMotionObjectDetector.from_dict(
        {
            "enabled": True,
            "roi": [0.25, 0.25, 0.75, 0.75],
            "min_area_px": 200,
            "motion_threshold": 10,
            "warmup_frames": 1,
            "skin_required": True,
            "skin_overlap_ratio_min": 0.30,
        }
    )
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    detector.detect(CameraPacket(rgb=base.copy(), timestamp=0.0))

    moved = base.copy()
    # Bright green block: lots of motion, no skin overlap
    moved[40:90, 50:120] = (0, 255, 0)
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.1))
    assert objects == []
    assert detector.last_triggered is False
    assert detector.last_motion_area > 0
    assert detector.last_skin_overlap_ratio < 0.30

def test_skin_motion_can_disable_skin_check():
    detector = SkinMotionObjectDetector.from_dict(
        {
            "enabled": True,
            "roi": [0.25, 0.25, 0.75, 0.75],
            "min_area_px": 200,
            "motion_threshold": 10,
            "warmup_frames": 1,
            "skin_required": False,
        }
    )
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    detector.detect(CameraPacket(rgb=base.copy(), timestamp=0.0))

    moved = base.copy()
    moved[40:90, 50:120] = (0, 255, 0)
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.1))
    assert len(objects) == 1
    assert objects[0].label == "motion_skin"  # honest default label since 2026-06-22 semantic-path change

