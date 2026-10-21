"""Multi-target separation tests (2026-06-22, 博's "complex scene" concern).

The hard real-world question: when MORE THAN ONE thing is in the scene at once,
does the system correctly STOP for the dangerous one while still exempting the
authorized cart? A safety system that only works with a single clean target is
not trustworthy on a real cabinet.

Scenes:
  A  authorized cart body + a hand together -> STOP (hand wins, cart not a shield)
  B  authorized cart body alone in cart region -> may proceed (exempted)
  C  cart body + unknown obstacle off to the side -> STOP (unknown not exempted)
  D  two objects both human -> STOP (highest priority)
  E  a hand sitting INSIDE the cart region must STILL stop (cart exemption never
     covers a human/limb label)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

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
    return ObjectObservation(label=label, pose=Pose(x=x, y=0.4, z=z))

def test_cart_body_plus_hand_stops_for_hand():
    """Scene A: authorized cart present AND a hand intrudes -> must STOP.
    The cart's presence must NOT shield the hand from triggering."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[
            _obj("motion", x=0.0),   # cart body near marker (would be exempt alone)
            _obj("hand", x=0.1),     # a hand also present
        ],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE.value

def test_cart_body_alone_may_proceed():
    """Scene B: only the authorized cart body in the cart region -> not a STOP
    from that body (exempted). Establishes the contrast for scene A/C."""
    now = [100.0]
    engine = FusionDecisionEngine(base_config("indoor"))
    import safezone_fusion.src.waiting_window as ww
    orig = ww.time.time
    try:
        ww.time.time = lambda: now[0]
        engine.decide(SensorFrame(marker=marker(center=(320, 240), jitter=0.5),
                                  objects=[_obj("motion", x=0.0)]))
        now[0] = 101.0
        result = engine.decide(SensorFrame(marker=marker(center=(321, 240), jitter=1.0),
                                           objects=[_obj("motion", x=0.0)]))
    finally:
        ww.time.time = orig
    assert result.safety.stop is False
    assert result.decision == Decision.ALLOW_CART

def test_cart_body_plus_unknown_obstacle_stops():
    """Scene C: cart body in region + an unknown obstacle off to the side ->
    STOP for the unknown (not exempted), reason unknown-obstacle not limb."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[
            _obj("motion", x=0.0),    # cart body (exempt)
            _obj("motion", x=0.5),    # unknown obstacle far from marker -> not exempt
        ],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE.value

def test_two_humans_stop():
    """Scene D: two human-labelled objects -> STOP (human priority)."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[_obj("person", x=-0.2), _obj("hand", x=0.2)],
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.HUMAN_IN_STOP_ZONE.value

def test_hand_inside_cart_region_still_stops():
    """Scene E: a hand located right where the cart body is (inside the cart
    region) must STILL stop -- the cart exemption never covers a human/limb."""
    engine = FusionDecisionEngine(base_config("indoor"))
    frame = SensorFrame(
        marker=marker(center=(320, 240), jitter=0.5),
        objects=[_obj("hand", x=0.0)],  # exactly at the marker/cart region
    )
    result = engine.decide(frame)
    assert result.decision == Decision.STOP
    assert result.reason == StopReason.LIMB_IN_STOP_ZONE.value
