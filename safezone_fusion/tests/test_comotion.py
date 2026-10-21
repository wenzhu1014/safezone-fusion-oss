from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.comotion import (
    CoMotionConfig,
    MotionTracker,
    is_co_moving,
)

# ---- MotionTracker ----

def test_tracker_basic_velocity():
    t = MotionTracker()
    t.update((0.0, 0.0))
    t.update((5.0, 0.0))
    t.update((10.0, 0.0))
    vx, vy = t.velocity
    assert vx == pytest.approx(5.0, abs=0.5)
    assert vy == pytest.approx(0.0, abs=0.5)

def test_tracker_coasts_on_none():
    t = MotionTracker(coast_budget=3)
    t.update((0.0, 0.0))
    t.update((4.0, 0.0))
    v_before = t.velocity
    # lose the point briefly: velocity is held
    held = t.update(None)
    assert held == v_before
    # after coast budget, velocity decays to zero
    t.update(None); t.update(None); t.update(None)
    assert t.velocity == (0.0, 0.0)

# ---- co-motion decision ----

def test_cart_body_moves_with_marker_is_exempt():
    """博's case: car body moving same direction/speed as marker -> co-moving."""
    cfg = CoMotionConfig()
    marker_v = (6.0, 0.0)   # marker moving right
    body_v = (5.5, 0.3)     # body moving right, ~same
    r = is_co_moving(marker_v, body_v, cfg)
    assert r.decidable is True
    assert r.co_moving is True

def test_hand_from_other_direction_not_exempt():
    """A hand intruding from another direction does NOT move with the marker."""
    cfg = CoMotionConfig()
    marker_v = (6.0, 0.0)    # marker moving right
    hand_v = (0.0, 6.0)      # hand coming down from top -> 90 deg different
    r = is_co_moving(marker_v, hand_v, cfg)
    assert r.decidable is True
    assert r.co_moving is False
    assert "diff_direction" in r.reason

def test_opposite_direction_not_exempt():
    cfg = CoMotionConfig()
    r = is_co_moving((6.0, 0.0), (-6.0, 0.0), cfg)
    assert r.co_moving is False

def test_same_direction_but_very_different_speed_not_exempt():
    cfg = CoMotionConfig()
    # same direction but 4x faster -> not rigidly attached
    r = is_co_moving((3.0, 0.0), (12.0, 0.0), cfg)
    assert r.decidable is True
    assert r.co_moving is False
    assert "diff_speed" in r.reason

def test_stopped_marker_is_not_decidable():
    """Cart stopped -> no motion vector -> must fall back to position exemption."""
    cfg = CoMotionConfig(min_speed_px=1.5)
    r = is_co_moving((0.2, 0.1), (0.0, 0.0), cfg)
    assert r.decidable is False
    assert r.co_moving is False
    assert "too_slow" in r.reason

def test_slight_angle_within_tolerance_is_co_moving():
    cfg = CoMotionConfig(max_angle_deg=25.0)
    # ~15 degrees off, similar speed
    import math
    speed = 6.0
    a = math.radians(15)
    r = is_co_moving((speed, 0.0), (speed * math.cos(a), speed * math.sin(a)), cfg)
    assert r.co_moving is True

def test_integration_tracker_plus_decision():
    """End-to-end: feed centroids for marker and body, then decide."""
    cfg = CoMotionConfig()
    mk = MotionTracker()
    bd = MotionTracker()
    # both slide right together over several frames
    for i in range(5):
        mk.update((float(i * 6), 100.0))
        bd.update((float(i * 6) + 20.0, 130.0))  # body offset but same motion
    r = is_co_moving(mk.velocity, bd.velocity, cfg)
    assert r.co_moving is True
