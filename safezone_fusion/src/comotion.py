from __future__ import annotations

"""Co-motion exemption: identify the cart body as "the thing moving together
with the marker".

博's idea (2026-06-22)
----------------------
The cart and its marker are rigidly attached: the marker and the car body move
as one. A hand intruding from the side does NOT move together with the marker.
So instead of carving out a fixed region around the marker (which can wrongly
exempt a hand that happens to be nearby, or fail to cover an oddly shaped car
body), we judge by MOTION CORRELATION:

    a moving blob whose velocity matches the marker's velocity
        -> rigidly attached to the marker -> cart body -> exempt
    a moving blob with a different velocity (e.g. a hand from another direction)
        -> independent intruder -> STOP

This is the "common fate" / rigid-motion grouping principle.

Important safety design
------------------------
* This exemption applies ONLY while the marker is authorized and inside the
  docking corridor (enforced by the caller). Motion correlation alone never
  exempts anything.
* When the cart is STOPPED (marker velocity ~0), there is no motion vector to
  compare, so co-motion cannot decide. The caller must fall back to the
  position-based exemption (marker-region) for a stopped, settled cart.
* Human/limb labels are never exempted regardless of co-motion (handled by the
  caller). This module only answers "does this blob move with the marker?".

Everything here is pure vector math so it is fully unit-testable without frames.
"""

from collections import deque
from dataclasses import dataclass, field
from math import hypot, atan2, degrees, cos, sin, radians
from typing import Deque, Optional, Tuple

Vec = Tuple[float, float]

def _smooth(samples) -> Vec:
    """Average a small set of (dx,dy) samples to denoise the velocity."""
    if not samples:
        return (0.0, 0.0)
    sx = sum(s[0] for s in samples) / len(samples)
    sy = sum(s[1] for s in samples) / len(samples)
    return (sx, sy)

@dataclass
class MotionTracker:
    """Tracks the velocity (px/frame) of a point from successive centroids.

    Feed the centroid each frame; reads back a smoothed velocity vector. Keeps a
    short window so a single noisy frame does not swing the vector. Tolerates a
    None update (point briefly lost) by holding the last velocity for a few
    frames (coasting), matching the marker-coasting principle elsewhere.
    """

    window: int = 5
    coast_budget: int = 4
    _pts: Deque[Optional[Vec]] = field(default_factory=lambda: deque(maxlen=6))
    _vels: Deque[Vec] = field(default_factory=lambda: deque(maxlen=5))
    _last_vel: Vec = (0.0, 0.0)
    _coast: int = 0
    _have_prev: bool = False
    _prev: Optional[Vec] = None

    def update(self, point: Optional[Vec]) -> Vec:
        if point is None:
            # coast: hold last velocity for a few frames
            if self._coast < self.coast_budget:
                self._coast += 1
                return self._last_vel
            self._have_prev = False
            self._prev = None
            self._last_vel = (0.0, 0.0)
            return self._last_vel

        self._coast = 0
        if self._have_prev and self._prev is not None:
            dx = point[0] - self._prev[0]
            dy = point[1] - self._prev[1]
            self._vels.append((dx, dy))
            self._last_vel = _smooth(list(self._vels))
        self._prev = point
        self._have_prev = True
        return self._last_vel

    @property
    def velocity(self) -> Vec:
        return self._last_vel

    @property
    def speed(self) -> float:
        return hypot(*self._last_vel)

@dataclass
class CoMotionConfig:
    """Thresholds for deciding whether two velocity vectors 'move together'."""

    # Below this speed (px/frame) the marker is considered STOPPED -> co-motion
    # cannot decide; caller should use position-based exemption.
    min_speed_px: float = 1.5
    # Max angle difference (deg) between blob and marker velocity to be "same
    # direction".
    max_angle_deg: float = 25.0
    # Blob speed must be within this ratio band of the marker speed
    # (rigid attachment -> similar speed).
    speed_ratio_min: float = 0.5
    speed_ratio_max: float = 2.0

def _angle_between(a: Vec, b: Vec) -> float:
    """Absolute angle difference in degrees between two vectors (0..180)."""
    aa = degrees(atan2(a[1], a[0]))
    ab = degrees(atan2(b[1], b[0]))
    d = abs(aa - ab) % 360.0
    return d if d <= 180.0 else 360.0 - d

@dataclass
class CoMotionResult:
    co_moving: bool
    decidable: bool          # False when marker is too slow to judge by motion
    angle_deg: float
    speed_ratio: float
    reason: str

def is_co_moving(marker_vel: Vec, object_vel: Vec, cfg: CoMotionConfig) -> CoMotionResult:
    """Decide if ``object_vel`` moves together with ``marker_vel``.

    Returns decidable=False when the marker is essentially stationary (the
    caller must then use the position-based, settled-cart exemption instead).
    """
    m_speed = hypot(*marker_vel)
    o_speed = hypot(*object_vel)

    if m_speed < cfg.min_speed_px:
        return CoMotionResult(False, False, 0.0, 0.0,
                              f"marker_too_slow:{m_speed:.2f}<{cfg.min_speed_px}")

    angle = _angle_between(marker_vel, object_vel)
    ratio = (o_speed / m_speed) if m_speed > 0 else 0.0

    same_dir = angle <= cfg.max_angle_deg
    same_speed = cfg.speed_ratio_min <= ratio <= cfg.speed_ratio_max
    co = same_dir and same_speed
    if co:
        reason = f"co_moving:angle={angle:.0f},ratio={ratio:.2f}"
    elif not same_dir:
        reason = f"diff_direction:angle={angle:.0f}>{cfg.max_angle_deg:.0f}"
    else:
        reason = f"diff_speed:ratio={ratio:.2f}"
    return CoMotionResult(co, True, angle, ratio, reason)
