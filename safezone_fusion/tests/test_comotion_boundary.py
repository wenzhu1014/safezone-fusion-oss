"""Co-motion exemption boundary / adversarial tests (2026-06-22).

博's idea (exempt the thing moving with the marker) has a shadow side: what if
a HAND moves together with the cart? e.g. a person pushing/steadying the cart
walks alongside it at the same speed. We must understand and bound this.

Key safety principle established here:
  Co-motion exemption operates only on the MOTION-net blob in roi_motion. The
  SEMANTIC hand layer (mediapipe_hand) is independent and is NEVER co-motion
  exempted -- so a recognized hand still STOPs even if it moves with the cart.
  Co-motion only decides whether a generic motion blob is the cart body; it can
  never turn a 'hand' label into 'allowed'.

These tests pin that contract at the comotion logic level and document the
residual case (an UNRECOGNIZED limb moving in perfect sync) as a known bound,
covered by the independent semantic layer + safety bias.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.comotion import CoMotionConfig, is_co_moving

def test_hand_pushing_cart_same_motion_is_geometrically_comoving():
    """Honest documentation: a hand moving at the SAME velocity as the marker is,
    by pure motion, indistinguishable from the cart body. is_co_moving returns
    True here -- which is exactly why the motion layer must NOT be the only
    guard. The semantic hand layer (independent, never exempted) is what stops a
    recognized hand in this case."""
    cfg = CoMotionConfig()
    r = is_co_moving((6.0, 0.0), (6.0, 0.0), cfg)
    assert r.co_moving is True  # geometrically identical motion
    # -> relies on semantic layer + the fact a real hand reaching IN typically
    #    has a different (approach) vector; see next tests.

def test_hand_reaching_in_while_cart_moves_is_not_comoving():
    """Realistic case: the cart slides laterally (marker moves sideways) while a
    hand reaches IN toward the cabinet (different direction). Not co-moving."""
    cfg = CoMotionConfig()
    # cart moving right; hand approaching downward/inward
    r = is_co_moving((6.0, 0.0), (1.0, 6.0), cfg)
    assert r.co_moving is False
    assert "diff_direction" in r.reason

def test_hand_faster_than_cart_not_comoving():
    """A reaching hand is usually faster/jerkier than the slow cart approach."""
    cfg = CoMotionConfig()
    r = is_co_moving((3.0, 0.0), (9.0, 0.0), cfg)  # 3x speed
    assert r.co_moving is False
    assert "diff_speed" in r.reason

def test_tightened_angle_tolerance_rejects_near_sync():
    """Operators can tighten max_angle_deg to be stricter about what counts as
    'moving together', shrinking the adversarial window."""
    strict = CoMotionConfig(max_angle_deg=10.0)
    # 18 degrees off: rejected under strict, would pass under default 25
    speed = 6.0
    a = math.radians(18)
    v = (speed * math.cos(a), speed * math.sin(a))
    assert is_co_moving((speed, 0.0), v, strict).co_moving is False
    assert is_co_moving((speed, 0.0), v, CoMotionConfig(max_angle_deg=25.0)).co_moving is True

def test_tightened_speed_band_rejects_mismatch():
    strict = CoMotionConfig(speed_ratio_min=0.8, speed_ratio_max=1.25)
    # 1.5x speed: rejected under strict band
    assert is_co_moving((4.0, 0.0), (6.0, 0.0), strict).co_moving is False

def test_stopped_cart_never_comotion_exempts_a_hand():
    """When the cart is stopped, co-motion is undecidable -> it can never exempt
    anything (including a hand). The settled cart uses position exemption, and a
    hand in the stop zone is caught by the semantic layer."""
    cfg = CoMotionConfig()
    r = is_co_moving((0.0, 0.0), (0.0, 0.0), cfg)  # both stationary
    assert r.decidable is False
    assert r.co_moving is False
