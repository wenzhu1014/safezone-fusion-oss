"""Layered (multi-detector) VisionSensorFusion tests (2026-06-22).

Closes the "bare forearm" safety gap found during 2026-06-22 real-camera
testing: mediapipe_hand only recognises hands/fingers, so a bare forearm (no
palm visible) produces NO detection. Running a motion safety-net (roi_motion)
ALONGSIDE the semantic detector catches that motion.

These tests use fake detectors to keep the test hermetic (no camera, no
mediapipe model download): we only verify the fusion *merging* contract and the
downstream safety prioritisation of the merged labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.models import ObjectObservation, Pose, StopReason
from safezone_fusion.src.safety_zone import SafetyZoneMonitor
from safezone_fusion.src.vision.interfaces import (
    CameraPacket,
    PirPacket,
    VisionSensorFusion,
)

class _NoMarker:
    last_corners = None

    def detect(self, packet):
        return None

class _NoPir:
    def read(self):
        return PirPacket(triggered=False)

class _FakeDetector:
    """Emits a fixed list of objects, ignoring the packet."""

    def __init__(self, objects):
        self._objects = objects
        self.drawn = False

    def detect(self, packet):
        return list(self._objects)

    def draw_debug(self, frame):
        self.drawn = True
        return frame

def _obj(label: str, x: float = 0.0) -> ObjectObservation:
    return ObjectObservation(label=label, pose=Pose(x=x, y=0.4, z=0.4))

def test_single_detector_still_works_back_compat():
    """A single (non-list) detector must behave exactly as before."""
    det = _FakeDetector([_obj("hand")])
    fusion = VisionSensorFusion(_NoMarker(), det, _NoPir())
    frame = fusion.read(CameraPacket(rgb=object()))
    assert [o.label for o in frame.objects] == ["hand"]
    # back-compat accessor
    assert fusion.object_detector is det

def test_layered_merges_objects_from_all_detectors():
    """Two detectors -> their objects are merged into one frame."""
    semantic = _FakeDetector([])              # mediapipe sees no hand (bare forearm)
    motion = _FakeDetector([_obj("motion")])  # motion net sees the moving forearm
    fusion = VisionSensorFusion(_NoMarker(), [semantic, motion], _NoPir())
    frame = fusion.read(CameraPacket(rgb=object()))
    assert [o.label for o in frame.objects] == ["motion"]

def test_bare_forearm_still_stops_via_motion_net():
    """The whole point: hand model misses the bare forearm, but the motion net
    catches it -> safety still STOPs (as an unknown obstacle, honestly)."""
    semantic = _FakeDetector([])              # no hand detected
    motion = _FakeDetector([_obj("motion")])  # forearm motion detected
    fusion = VisionSensorFusion(_NoMarker(), [semantic, motion], _NoPir())
    frame = fusion.read(CameraPacket(rgb=object()))

    monitor = SafetyZoneMonitor({
        "safety": {
            "stop_zone": {"x_min": -0.5, "x_max": 0.5, "y_min": 0.2, "y_max": 1.8, "z_min": 0.0, "z_max": 0.8},
            "warning_zone": {"x_min": -0.75, "x_max": 0.75, "y_min": 0.1, "y_max": 1.9, "z_min": 0.0, "z_max": 1.1},
            "pir_stop_when_unknown_in_stop_zone": True,
        }
    })
    result = monitor.check(frame)
    assert result.stop is True
    assert result.reason == StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE

def test_layered_hand_wins_over_motion_when_both_fire():
    """When the hand IS visible, both detectors fire; the limb reason wins."""
    semantic = _FakeDetector([_obj("hand")])
    motion = _FakeDetector([_obj("motion")])
    fusion = VisionSensorFusion(_NoMarker(), [semantic, motion], _NoPir())
    frame = fusion.read(CameraPacket(rgb=object()))
    assert {o.label for o in frame.objects} == {"hand", "motion"}

    monitor = SafetyZoneMonitor({
        "safety": {
            "stop_zone": {"x_min": -0.5, "x_max": 0.5, "y_min": 0.2, "y_max": 1.8, "z_min": 0.0, "z_max": 0.8},
            "warning_zone": {"x_min": -0.75, "x_max": 0.75, "y_min": 0.1, "y_max": 1.9, "z_min": 0.0, "z_max": 1.1},
            "pir_stop_when_unknown_in_stop_zone": True,
        }
    })
    result = monitor.check(frame)
    assert result.stop is True
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE

def test_draw_debug_runs_on_all_detectors():
    a = _FakeDetector([])
    b = _FakeDetector([])
    fusion = VisionSensorFusion(_NoMarker(), [a, b], _NoPir())
    fusion.draw_debug(frame=object())
    assert a.drawn is True
    assert b.drawn is True
