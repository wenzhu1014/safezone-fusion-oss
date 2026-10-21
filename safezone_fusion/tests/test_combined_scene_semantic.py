"""End-to-end combined-scene tests for the semantic detection path (2026-06-22).

These tests exercise the FULL FusionDecisionEngine (not just SafetyZoneMonitor)
to prove the *value* of the semantic path in realistic combined scenes:

Scene A  hand in front of marker      -> STOP, reason=limb, voice=stop_human
Scene B  paper drifts (semantic only) -> mediapipe sees no hand -> NO object ->
                                          cart may proceed (no false STOP)
Scene C  paper drifts (motion net)    -> roi_motion emits 'motion' -> STOP, but
                                          reason=unknown_obstacle, voice=stop_obstacle
                                          (still safe, but does NOT lie about a limb)

The contrast between B and C is the whole point: a semantic detector does not
false-trigger on non-hand motion, while the motion safety-net still stops but
reports an *honest* reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import safezone_fusion.src.waiting_window as waiting_window_module
from safezone_fusion.src.fusion import FusionDecisionEngine
from safezone_fusion.src.models import (
    Decision,
    ObjectObservation,
    Pose,
    SensorFrame,
    StopReason,
)

from test_acceptance_v084_synthetic import base_config, marker

def _obj(label: str, *, x: float = 0.0, z: float = 0.4) -> ObjectObservation:
    """An object squarely inside the stop_zone."""
    return ObjectObservation(label=label, pose=Pose(x=x, y=0.4, z=z))

def test_scene_hand_in_front_of_marker_stops_as_human():
    """Scene A: a confirmed hand (mediapipe 'hand') in front of an authorized,
    stable cart marker must STOP with a human/limb reason and the human voice
    prompt -- safety overrides the otherwise-allowed docking."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[_obj("hand")],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE.value
    assert result.voice == base_config("indoor")["voice"]["stop_human"]

def test_scene_paper_drift_semantic_only_does_not_false_stop():
    """Scene B: with a semantic detector, drifting paper produces NO hand object
    at all (mediapipe simply doesn't detect a hand). The engine therefore does
    not STOP on the non-hand motion. We model 'no detection' as an empty objects
    list -- the semantic detector's correct output for paper."""
    now = [100.0]
    engine = FusionDecisionEngine(base_config("indoor"))
    # Prime + stabilise an authorized cart in the waiting window.
    import safezone_fusion.src.waiting_window as ww
    orig_time = ww.time.time
    try:
        ww.time.time = lambda: now[0]
        engine.decide(SensorFrame(marker=marker(center=(320, 240), jitter=0.5), objects=[]))
        now[0] = 101.0
        result = engine.decide(
            SensorFrame(marker=marker(center=(322, 240), jitter=1.0), objects=[])
        )
    finally:
        ww.time.time = orig_time
    # No hand detected -> no STOP from a phantom limb. Cart is allowed to proceed.
    assert result.decision == Decision.ALLOW_CART
    assert result.safety.stop is False

def test_scene_paper_drift_motion_net_stops_as_unknown_not_limb():
    """Scene C: with the motion safety-net (roi_motion), drifting paper emits an
    honest 'motion' object. The engine STILL stops (safety-net preserved) but the
    reason is UNKNOWN_OBSTACLE and the voice is the obstacle prompt -- it must NOT
    falsely claim a human limb."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        # Off to the side so it is NOT exempted as the cart body
        # (verified_cart_region x_radius=0.30 around the marker at x=0).
        objects=[_obj("motion", x=0.4)],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE.value
    assert result.reason != StopReason.LIMB_IN_STOP_ZONE.value
    assert result.voice == base_config("indoor")["voice"]["stop_obstacle"]

def test_scene_hand_beats_motion_when_both_present():
    """If both a confirmed hand and generic motion are in the stop zone, the
    human/limb reason wins (highest safety priority)."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[_obj("motion"), _obj("hand")],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE.value
