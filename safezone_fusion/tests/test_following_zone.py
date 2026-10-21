from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.following_zone import (
    FaceToPixelCalibration,
    FollowingZoneConfig,
    compute_following_zone,
)

def _area(box):
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)

def test_zone_follows_device_position():
    cfg = FollowingZoneConfig()
    left = compute_following_zone(0.2, 0.3, False, cfg)
    right = compute_following_zone(0.8, 0.7, False, cfg)
    # danger box center should track the device center
    lcx = (left.danger_box[0] + left.danger_box[2]) / 2
    rcx = (right.danger_box[0] + right.danger_box[2]) / 2
    assert lcx < 0.4 < rcx
    assert left.known and right.known

def test_zone_stays_small_not_whole_face():
    # 博's worry: the box must NOT cover the whole face
    cfg = FollowingZoneConfig(device_half_w=0.08, device_half_h=0.08, margin=0.06)
    r = compute_following_zone(0.5, 0.5, False, cfg)
    # outer half-size = 0.14 -> box is 0.28 x 0.28 = ~0.078 of the face
    assert _area(r.danger_box) < 0.10  # well under 10% of the face
    assert _area(r.danger_box) > 0.0

def test_device_body_is_excluded():
    cfg = FollowingZoneConfig()
    r = compute_following_zone(0.5, 0.5, False, cfg)
    # device_box (to exclude) is strictly inside danger_box (to watch)
    dx0, dy0, dx1, dy1 = r.device_box
    gx0, gy0, gx1, gy1 = r.danger_box
    assert gx0 <= dx0 and gy0 <= dy0 and dx1 <= gx1 and dy1 <= gy1
    assert _area(r.device_box) < _area(r.danger_box)

def test_moving_inflates_zone():
    cfg = FollowingZoneConfig(inflate_when_moving=0.03)
    still = compute_following_zone(0.5, 0.5, False, cfg)
    moving = compute_following_zone(0.5, 0.5, True, cfg)
    assert _area(moving.danger_box) > _area(still.danger_box)
    assert "moving" in moving.reason

def test_unknown_pose_falls_back_to_box():
    cfg = FollowingZoneConfig(whole_face_when_unknown=True, fallback_box=(0.3, 0.5, 0.7, 0.95))
    r = compute_following_zone(None, None, False, cfg)
    assert r.known is False
    assert r.danger_box == (0.3, 0.5, 0.7, 0.95)
    assert "fallback" in r.reason

def test_unknown_pose_no_zone_when_configured():
    cfg = FollowingZoneConfig(whole_face_when_unknown=False)
    r = compute_following_zone(None, None, False, cfg)
    assert r.danger_box is None
    assert r.device_box is None

def test_zone_clamped_at_face_edges():
    cfg = FollowingZoneConfig(device_half_w=0.08, device_half_h=0.08, margin=0.06)
    # device near top-left corner: box must clamp to [0,1], not go negative
    r = compute_following_zone(0.02, 0.02, False, cfg)
    x0, y0, x1, y1 = r.danger_box
    assert x0 >= 0.0 and y0 >= 0.0
    assert x1 <= 1.0 and y1 <= 1.0

def test_calibration_box_to_pixels():
    calib = FaceToPixelCalibration.identity(1280, 720)
    px = calib.box_to_pixels((0.0, 0.0, 1.0, 1.0), 1280, 720)
    assert px == (0, 0, 1280, 720)
    px2 = calib.box_to_pixels((0.25, 0.5, 0.75, 1.0), 1280, 720)
    assert px2 == (320, 360, 960, 720)

def test_calibration_with_offset_scale():
    # face occupies pixels x:[100,500], y:[50,450]
    calib = FaceToPixelCalibration(sx=400, sy=400, ox=100, oy=50)
    px = calib.box_to_pixels((0.0, 0.0, 1.0, 1.0), 1280, 720)
    assert px == (100, 50, 500, 450)
