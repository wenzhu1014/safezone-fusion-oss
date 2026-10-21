from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.geometry import in_image_corridor, in_waiting_window
from safezone_fusion.src.models import MarkerObservation, ObjectObservation, Pose, SensorFrame
from safezone_fusion.src.safety_zone import SafetyZoneMonitor

# ---- image-corridor geometry ----

def test_in_image_corridor_inside():
    corridor = {"x_min": 0.3, "y_min": 0.3, "x_max": 0.7, "y_max": 0.7}
    assert in_image_corridor((640, 360), corridor, 1280, 720) is True  # center

def test_in_image_corridor_outside():
    corridor = {"x_min": 0.3, "y_min": 0.3, "x_max": 0.7, "y_max": 0.7}
    assert in_image_corridor((100, 360), corridor, 1280, 720) is False  # far left

def test_in_image_corridor_unknown_center_is_false():
    corridor = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    assert in_image_corridor(None, corridor, 1280, 720) is False

# ---- SafetyZoneMonitor with image-corridor mode ----

def _config(corridor_mode):
    corridor = {"mode": corridor_mode, "x_min": 0.3, "y_min": 0.3, "x_max": 0.7, "y_max": 0.7,
                # pose-mode fields (used only when mode=pose)
                "x_min_": -0.3}
    if corridor_mode == "pose":
        corridor = {"mode": "pose", "x_min": -0.3, "x_max": 0.3,
                    "distance_min": 0.2, "distance_max": 0.7, "yaw_abs_max_deg": 10.0}
    return {
        "safety": {
            "stop_zone": {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": 0, "z_max": 2},
            "warning_zone": {"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2, "z_min": 0, "z_max": 3},
            "pir_stop_when_unknown_in_stop_zone": True,
        },
        "verified_cart_region": {"x_radius": 0.3, "y_radius": 0.3, "z_radius": 0.3},
        "docking_corridor": corridor,
        "camera": {"width": 1280, "height": 720},
    }

def _frame_with_marker_and_cart(center_px):
    marker = MarkerObservation(marker_id=1, pose=Pose(0.0, 0.0, 0.4), stable_frames=10,
                               center_px=center_px)
    # a cart-body object near the marker (same pose)
    cart_obj = ObjectObservation(label="motion", pose=Pose(0.0, 0.0, 0.4), confidence=0.8)
    return SensorFrame(marker=marker, objects=[cart_obj])

def test_image_mode_exempts_cart_when_marker_in_corridor():
    mon = SafetyZoneMonitor(_config("image"))
    frame = _frame_with_marker_and_cart((640, 360))  # center -> in corridor
    result = mon.check(frame, cart_verified=True)
    # cart body should be exempted -> no stop
    assert result.stop is False

def test_image_mode_does_not_exempt_when_marker_outside_corridor():
    mon = SafetyZoneMonitor(_config("image"))
    frame = _frame_with_marker_and_cart((100, 360))  # far left -> outside corridor
    result = mon.check(frame, cart_verified=True)
    # not in corridor -> cart not exempted -> unknown obstacle -> stop
    assert result.stop is True

def test_image_mode_human_never_exempted():
    mon = SafetyZoneMonitor(_config("image"))
    marker = MarkerObservation(marker_id=1, pose=Pose(0.0, 0.0, 0.4), stable_frames=10,
                               center_px=(640, 360))
    human = ObjectObservation(label="person", pose=Pose(0.0, 0.0, 0.4), confidence=0.9)
    frame = SensorFrame(marker=marker, objects=[human])
    result = mon.check(frame, cart_verified=True)
    assert result.stop is True  # human always stops, even in corridor

def test_pose_mode_still_works():
    mon = SafetyZoneMonitor(_config("pose"))
    # marker at distance 0.4, centered, facing -> in pose corridor
    frame = _frame_with_marker_and_cart((640, 360))
    result = mon.check(frame, cart_verified=True)
    assert result.stop is False
