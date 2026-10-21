from __future__ import annotations

import numpy as np

from safezone_fusion.src.vision.interfaces import CameraPacket
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector

def test_roi_motion_detector_disabled_emits_no_objects():
    detector = RoiMotionObjectDetector.from_dict({"enabled": False})
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detector.detect(CameraPacket(rgb=frame, timestamp=0.0)) == []

def test_roi_motion_detector_triggers_on_motion_inside_roi():
    detector = RoiMotionObjectDetector.from_dict(
        {
            "enabled": True,
            "roi": [0.25, 0.25, 0.75, 0.75],
            "min_area_px": 80,
            "threshold": 10,
            "warmup_frames": 1,
            "min_consecutive": 3,
            "label": "hand",
        }
    )
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detector.detect(CameraPacket(rgb=base.copy(), timestamp=0.0)) == []

    # Need 3 consecutive motion frames to trigger
    moved = base.copy()
    moved[50:80, 70:100] = 255
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.1))
    assert len(objects) == 0  # 1 consecutive, not enough
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.2))
    assert len(objects) == 0  # 2 consecutive, not enough
    objects = detector.detect(CameraPacket(rgb=moved, timestamp=0.3))
    assert len(objects) == 1  # 3 consecutive, triggered
    assert objects[0].label == "hand"
    assert detector.last_triggered is True
    assert detector.last_motion_area >= 80

def test_roi_motion_detector_ignores_motion_outside_roi():
    detector = RoiMotionObjectDetector.from_dict(
        {
            "enabled": True,
            "roi": [0.4, 0.4, 0.6, 0.6],
            "min_area_px": 80,
            "threshold": 10,
            "warmup_frames": 1,
            "label": "hand",
        }
    )
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    detector.detect(CameraPacket(rgb=base.copy(), timestamp=0.0))

    moved = base.copy()
    moved[0:30, 0:30] = 255
    assert detector.detect(CameraPacket(rgb=moved, timestamp=0.1)) == []
    assert detector.last_triggered is False
