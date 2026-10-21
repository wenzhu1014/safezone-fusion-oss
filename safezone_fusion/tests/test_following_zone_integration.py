from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector, RoiMotionConfig
from safezone_fusion.src.vision.interfaces import CameraPacket
from safezone_fusion.src.following_zone import (
    FollowingZoneConfig,
    FaceToPixelCalibration,
    compute_following_zone,
)

W, H = 320, 240

def _blank():
    return np.full((H, W, 3), 40, dtype=np.uint8)

def _with_block(frame, box_px, value=200):
    x0, y0, x1, y1 = box_px
    f = frame.copy()
    f[y0:y1, x0:x1] = value
    return f

def _make_detector():
    cfg = RoiMotionConfig(enabled=True, roi=(0.0, 0.0, 1.0, 1.0), min_area_px=200.0,
                          warmup_frames=3, min_consecutive=2, threshold=25)
    return RoiMotionObjectDetector(cfg)

def _run(det, frame, n=1):
    obs = []
    for _ in range(n):
        obs = det.detect(CameraPacket(rgb=frame))
    return obs

def test_dynamic_roi_follows_device_and_excludes_body():
    """博's two questions, end to end:
    1) the moving device body does NOT trigger STOP (it is masked)
    2) the watched ROI is the small following box, not the whole face
    """
    det = _make_detector()
    calib = FaceToPixelCalibration.identity(W, H)
    zcfg = FollowingZoneConfig(device_half_w=0.1, device_half_h=0.1, margin=0.06)

    # device at center, moving
    zone = compute_following_zone(0.5, 0.5, True, zcfg)
    det.set_dynamic_roi(zone.danger_box)
    dev_px = calib.box_to_pixels(zone.device_box, W, H)
    det.set_device_mask_px(dev_px)

    # warmup with an empty face
    _run(det, _blank(), n=4)

    # now the DEVICE BODY appears/moves inside its own mask region -> must NOT trigger
    frame_dev = _with_block(_blank(), dev_px, value=220)
    _run(det, frame_dev, n=3)
    assert det.last_triggered is False, "device body in its mask must not trigger STOP"

def test_hand_outside_device_but_in_danger_ring_triggers():
    det = _make_detector()
    calib = FaceToPixelCalibration.identity(W, H)
    # bigger margin so the ring is wide enough to clear the device-mask padding
    zcfg = FollowingZoneConfig(device_half_w=0.08, device_half_h=0.08, margin=0.18)

    zone = compute_following_zone(0.5, 0.5, True, zcfg)
    det.set_dynamic_roi(zone.danger_box)
    dev_px = calib.box_to_pixels(zone.device_box, W, H)
    det.set_device_mask_px(dev_px)
    _run(det, _blank(), n=4)

    # a hand enters the danger ring well above the device body (beyond mask pad)
    gx0, gy0, gx1, gy1 = calib.box_to_pixels(zone.danger_box, W, H)
    dx0, dy0, dx1, dy1 = dev_px
    hand_box = (dx0, gy0, dx1, dy0 - 16)  # strip above device, inside danger box
    assert hand_box[3] > hand_box[1], "ring must be wide enough for this test"
    frame_hand = _with_block(_blank(), hand_box, value=220)
    _run(det, frame_hand, n=3)
    assert det.last_triggered is True, "hand in danger ring (outside device) must trigger STOP"

def test_setting_none_reverts_to_static_roi():
    det = _make_detector()
    det.set_dynamic_roi((0.4, 0.4, 0.6, 0.6))
    assert det._dynamic_roi == (0.4, 0.4, 0.6, 0.6)
    det.set_dynamic_roi(None)
    assert det._dynamic_roi is None

def test_roi_shift_resets_background():
    det = _make_detector()
    det.set_dynamic_roi((0.4, 0.4, 0.6, 0.6))
    _run(det, _blank(), n=5)
    assert det._background is not None
    # a big shift should reset the background
    det.set_dynamic_roi((0.0, 0.0, 0.3, 0.3))
    assert det._background is None
