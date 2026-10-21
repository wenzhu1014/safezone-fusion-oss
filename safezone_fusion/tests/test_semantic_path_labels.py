"""Semantic detection path tests (2026-06-22).

Verifies the label-tier contract in SafetyZoneMonitor that underpins the
"semantic path" change:

- Motion-only detectors (roi_motion/skin_motion) now emit honest labels
  ('motion' / 'motion_skin') instead of falsely claiming 'hand'. They must
  STILL trigger a STOP, but via the UNKNOWN_OBSTACLE path, never LIMB.
- Only a semantic detector (mediapipe_hand) emits 'hand', which routes to the
  LIMB_IN_STOP_ZONE reason.
- This keeps the motion safety-net (never miss a stop) while making labels
  honest (don't lie about confirming a human limb).
"""

from __future__ import annotations

from safezone_fusion.src.models import (
    ObjectObservation,
    Pose,
    SensorFrame,
    StopReason,
)
from safezone_fusion.src.safety_zone import SafetyZoneMonitor

def _config():
    return {
        "safety": {
            "stop_zone": {
                "x_min": -0.5, "x_max": 0.5,
                "y_min": 0.2, "y_max": 1.8,
                "z_min": 0.0, "z_max": 0.8,
            },
            "warning_zone": {
                "x_min": -0.75, "x_max": 0.75,
                "y_min": 0.1, "y_max": 1.9,
                "z_min": 0.0, "z_max": 1.1,
            },
            "pir_stop_when_unknown_in_stop_zone": True,
        },
    }

def _obj(label: str) -> ObjectObservation:
    # Pose squarely inside the stop_zone.
    return ObjectObservation(label=label, pose=Pose(x=0.0, y=0.4, z=0.4))

def test_motion_label_stops_as_unknown_obstacle_not_limb():
    """roi_motion's honest 'motion' label must STILL stop, via UNKNOWN_OBSTACLE."""
    monitor = SafetyZoneMonitor(_config())
    result = monitor.check(SensorFrame(objects=[_obj("motion")]))
    assert result.stop is True  # safety-net preserved: still stops
    assert result.reason == StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE
    assert result.reason != StopReason.LIMB_IN_STOP_ZONE  # no false limb claim

def test_motion_skin_label_stops_as_unknown_obstacle():
    """skin_motion's honest 'motion_skin' label behaves the same as 'motion'."""
    monitor = SafetyZoneMonitor(_config())
    result = monitor.check(SensorFrame(objects=[_obj("motion_skin")]))
    assert result.stop is True
    assert result.reason == StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE

def test_hand_label_stops_as_limb():
    """mediapipe_hand's confirmed 'hand' label routes to the LIMB reason."""
    monitor = SafetyZoneMonitor(_config())
    result = monitor.check(SensorFrame(objects=[_obj("hand")]))
    assert result.stop is True
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE

def test_person_label_stops_as_human():
    """A confirmed 'person'/'human' label is the highest-priority human stop."""
    monitor = SafetyZoneMonitor(_config())
    result = monitor.check(SensorFrame(objects=[_obj("person")]))
    assert result.stop is True
    assert result.reason == StopReason.HUMAN_IN_STOP_ZONE

def test_no_objects_no_stop():
    monitor = SafetyZoneMonitor(_config())
    result = monitor.check(SensorFrame(objects=[]))
    assert result.stop is False
    assert result.reason == StopReason.NONE
