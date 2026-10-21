# -*- coding: utf-8 -*-
"""Safety-critical: a STATIC object left in the ROI must keep being detected
(occupancy semantics), not silently absorbed into the background (the old
motion-only blind spot)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector, RoiMotionConfig
from safezone_fusion.src.vision.interfaces import CameraPacket

def _empty(ts):
    return CameraPacket(rgb=np.full((480, 640, 3), 120, np.uint8), timestamp=ts)

def _with_object(ts):
    f = np.full((480, 640, 3), 120, np.uint8)
    # a solid block inside the default ROI (0.35..0.65 x, 0.35..0.85 y)
    f[200:360, 260:400] = 30
    return CameraPacket(rgb=f, timestamp=ts)

def _cfg():
    return RoiMotionConfig(enabled=True, warmup_frames=3, min_consecutive=2,
                           min_area_px=400.0, hold_static_occupant=True)

def test_static_object_keeps_being_detected():
    det = RoiMotionObjectDetector(_cfg())
    # warmup on empty zone
    for i in range(4):
        det.detect(_empty(float(i)))
    # place object; feed the SAME static frame many times
    triggered_history = []
    for i in range(40):
        objs = det.detect(_with_object(10.0 + i))
        triggered_history.append(len(objs) > 0)
    # it must remain detected the whole time, not fade out after a second
    assert triggered_history[2] is True, "should trigger shortly after placement"
    assert all(triggered_history[2:]), "static occupant must NOT fade into background"

def test_zone_clears_when_object_removed():
    det = RoiMotionObjectDetector(_cfg())
    for i in range(4):
        det.detect(_empty(float(i)))
    for i in range(10):
        det.detect(_with_object(10.0 + i))
    assert det.last_triggered is True
    # remove object -> zone should clear within a short time
    cleared = False
    for i in range(40):
        objs = det.detect(_empty(30.0 + i))
        if not objs:
            cleared = True
            break
    assert cleared, "zone must clear after the object is physically removed"

def test_pure_motion_mode_still_available():
    det = RoiMotionObjectDetector(RoiMotionConfig(enabled=True, warmup_frames=3,
                                  min_consecutive=2, min_area_px=400.0,
                                  hold_static_occupant=False))
    for i in range(4):
        det.detect(_empty(float(i)))
    # in pure-motion mode a static object eventually fades (old behaviour)
    faded = False
    for i in range(120):
        objs = det.detect(_with_object(10.0 + i))
        if not objs and i > 10:
            faded = True
            break
    assert faded, "pure-motion mode should still absorb a static object (opt-out works)"
