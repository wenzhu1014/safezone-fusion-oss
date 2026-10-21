# -*- coding: utf-8 -*-
"""Reproduce the real-machine flicker: an object FULLY inside the ROI is flat /
low-texture, so its diff area hovers near min_area_px and occasionally dips
below it. The occupancy latch + hysteresis must hold STOP steady instead of
flickering STOP->clear->STOP."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector, RoiMotionConfig
from safezone_fusion.src.vision.interfaces import CameraPacket

def _empty(ts):
    return CameraPacket(rgb=np.full((480, 640, 3), 120, np.uint8), timestamp=ts)

def _object(ts, fill_value):
    """Object covering the ROI. fill_value close to background (120) => small
    diff area (low texture); far => large area. Vary it to simulate a flat
    object whose diff hovers near threshold."""
    f = np.full((480, 640, 3), 120, np.uint8)
    f[180:400, 240:420] = fill_value
    return CameraPacket(rgb=f, timestamp=ts)

def test_flat_object_at_threshold_does_not_flicker():
    cfg = RoiMotionConfig(enabled=True, warmup_frames=3, min_consecutive=2,
                          min_area_px=4000.0, occupied_keep_area_px=150.0,
                          clear_frames_to_release=8, hold_static_occupant=True)
    det = RoiMotionObjectDetector(cfg)
    for i in range(4):
        det.detect(_empty(float(i)))
    # place a clearly-detectable object first (decisive contrast)
    for i in range(3):
        det.detect(_object(10.0 + i, 30))
    assert det.last_triggered is True

    # now the object stays but its measured diff hovers: some frames strong,
    # some frames weak (simulating flat surface + sensor noise dipping below
    # the ENTER threshold). It must stay latched (no flicker).
    history = []
    seq = [30, 118, 119, 30, 118, 30, 119, 118, 30, 30, 118, 119, 30, 30]
    for i, fv in enumerate(seq):
        objs = det.detect(_object(20.0 + i, fv))
        history.append(bool(objs))
    assert all(history), f"latched STOP should not flicker, got {history}"

def test_latch_releases_after_sustained_clear():
    cfg = RoiMotionConfig(enabled=True, warmup_frames=3, min_consecutive=2,
                          min_area_px=4000.0, occupied_keep_area_px=150.0,
                          clear_frames_to_release=8, hold_static_occupant=True)
    det = RoiMotionObjectDetector(cfg)
    for i in range(4):
        det.detect(_empty(float(i)))
    for i in range(5):
        det.detect(_object(10.0 + i, 30))
    assert det.last_triggered is True
    # sustained clear -> latch must release (zone genuinely empty)
    released = False
    for i in range(40):
        objs = det.detect(_empty(30.0 + i))
        if not objs:
            released = True
            break
    assert released
