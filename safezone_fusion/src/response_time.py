from __future__ import annotations

"""End-to-end response time measurement + ISO 13855 minimum safety distance.

Why this exists
---------------
ISO 13855:2024 ("Positioning of safeguards with respect to the approach of the
human body") makes the *total system response time* the central input for
computing the minimum safety (separation) distance:

    S = K * T + C

    S : minimum distance from the danger zone to the detection boundary (mm)
    K : approach speed of the body/limb (mm/s); ISO 13855 uses 2000 mm/s for a
        hand/arm at short range, 1600 mm/s for walking approach
    T : total response time (s) = T_detect (our camera->stop output) + T_machine
        (PLC reaction + servo/mechanical stopping)
    C : intrusion distance / approach allowance (mm); for detection that cannot
        discriminate a finger, ISO adds a term based on detection capability

Our subsystem (the camera board) only owns **T_detect**: the time from a hazard
becoming present in the frame to the moment we publish ``stop=true``. The PLC
team owns ``T_machine``. We must hand them a measured, worst-case T_detect so
they can finish the safety-distance calculation. This module:

* measures per-stage and end-to-end latency with a monotonic clock, and
* reports percentiles (mean / p50 / p95 / p99 / max) -- safety uses the worst
  case (p95/p99/max), never the average, and
* plugs T_detect into S = K*T + C so we can show the resulting safety distance.

The latency tracker has **no fusion dependency** so it can wrap any pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

# ---- approach speed constants from ISO 13855 (mm/s) ----
K_HAND_APPROACH = 2000.0   # hand/arm reaching, distance <= 500 mm regime
K_WALKING_APPROACH = 1600.0  # whole-body walking approach

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list. pct in [0,100]."""
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    # nearest-rank
    rank = max(1, int(round(pct / 100.0 * len(sorted_vals))))
    rank = min(rank, len(sorted_vals))
    return sorted_vals[rank - 1]

@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "min_ms": round(self.min_ms, 2),
        }

def summarize(samples_ms: List[float]) -> LatencyStats:
    if not samples_ms:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(samples_ms)
    return LatencyStats(
        count=len(s),
        mean_ms=sum(s) / len(s),
        p50_ms=_percentile(s, 50),
        p95_ms=_percentile(s, 95),
        p99_ms=_percentile(s, 99),
        max_ms=s[-1],
        min_ms=s[0],
    )

class ResponseTimer:
    """Measures per-frame end-to-end processing latency and per-stage splits.

    Usage per frame::

        timer.begin()                      # hazard present in this raw frame
        ... capture/convert ...
        timer.mark("capture")
        ... detection ...
        timer.mark("detect")
        ... fusion decision ...
        timer.mark("decide")
        ... emit safety signal ...
        timer.mark("emit")
        timer.end()                        # stop=true is now published

    ``end()`` records the total begin->end span. Stage marks record the split
    since the previous mark. Everything uses ``time.perf_counter`` (monotonic,
    high-resolution) so it is immune to wall-clock changes.
    """

    def __init__(self, clock=None):
        self._clock = clock or time.perf_counter
        self._t0: Optional[float] = None
        self._t_prev: Optional[float] = None
        self._totals_ms: List[float] = []
        self._stage_ms: Dict[str, List[float]] = {}

    def begin(self) -> None:
        now = self._clock()
        self._t0 = now
        self._t_prev = now

    def mark(self, stage: str) -> float:
        """Record elapsed since previous mark/begin for ``stage``. Returns ms."""
        if self._t_prev is None:
            self.begin()
        now = self._clock()
        dt_ms = (now - self._t_prev) * 1000.0
        self._stage_ms.setdefault(stage, []).append(dt_ms)
        self._t_prev = now
        return dt_ms

    def end(self) -> float:
        """Record total begin->now span. Returns ms for this frame."""
        if self._t0 is None:
            return 0.0
        total_ms = (self._clock() - self._t0) * 1000.0
        self._totals_ms.append(total_ms)
        self._t0 = None
        self._t_prev = None
        return total_ms

    # ---- reporting ----

    def total_stats(self) -> LatencyStats:
        return summarize(self._totals_ms)

    def stage_stats(self) -> Dict[str, LatencyStats]:
        return {name: summarize(vals) for name, vals in self._stage_ms.items()}

    def sample_count(self) -> int:
        return len(self._totals_ms)

    def report(self) -> Dict[str, object]:
        return {
            "total": self.total_stats().as_dict(),
            "stages": {k: v.as_dict() for k, v in self.stage_stats().items()},
        }

@dataclass
class SafetyDistanceResult:
    detect_time_s: float       # T_detect (our subsystem), worst-case
    machine_time_s: float      # T_machine (PLC + servo stop), supplied by integrator
    total_time_s: float        # T = T_detect + T_machine
    approach_speed_mm_s: float  # K
    intrusion_mm: float        # C
    min_distance_mm: float     # S = K*T + C

    def as_dict(self) -> Dict[str, float]:
        return {
            "detect_time_s": round(self.detect_time_s, 4),
            "machine_time_s": round(self.machine_time_s, 4),
            "total_time_s": round(self.total_time_s, 4),
            "approach_speed_mm_s": self.approach_speed_mm_s,
            "intrusion_mm": self.intrusion_mm,
            "min_distance_mm": round(self.min_distance_mm, 1),
        }

def min_safety_distance(
    detect_time_s: float,
    *,
    machine_time_s: float = 0.0,
    approach_speed_mm_s: float = K_HAND_APPROACH,
    intrusion_mm: float = 0.0,
) -> SafetyDistanceResult:
    """Compute ISO 13855 minimum separation distance S = K*(Td+Tm) + C.

    ``detect_time_s`` should be the *worst-case* camera-side response (e.g. p95
    or max of measured end-to-end latency, plus the detection debounce: any
    ``min_consecutive`` frames / enter_frames latch we require before STOP).
    ``machine_time_s`` is the PLC reaction + servo/mechanical stopping time,
    provided by the integrator. ``intrusion_mm`` (C) covers detection
    capability per ISO 13855 (e.g. limb vs whole body).
    """
    total = max(0.0, detect_time_s) + max(0.0, machine_time_s)
    s = approach_speed_mm_s * total + intrusion_mm
    return SafetyDistanceResult(
        detect_time_s=detect_time_s,
        machine_time_s=machine_time_s,
        total_time_s=total,
        approach_speed_mm_s=approach_speed_mm_s,
        intrusion_mm=intrusion_mm,
        min_distance_mm=s,
    )

def debounce_time_s(min_consecutive_frames: int, fps: float) -> float:
    """Latency added by requiring N consecutive detections before STOP.

    A safety detector that latches only after ``min_consecutive_frames`` frames
    adds (N-1)/fps seconds of worst-case delay on top of single-frame
    processing. This MUST be included in T_detect.
    """
    if fps <= 0 or min_consecutive_frames <= 1:
        return 0.0
    return (min_consecutive_frames - 1) / fps
