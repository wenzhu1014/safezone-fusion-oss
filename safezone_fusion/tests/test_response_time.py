from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.response_time import (
    K_HAND_APPROACH,
    K_WALKING_APPROACH,
    ResponseTimer,
    SafetyDistanceResult,
    debounce_time_s,
    min_safety_distance,
    summarize,
)

class FakeClock:
    """Deterministic monotonic clock driven by a list of timestamps (seconds)."""

    def __init__(self, times):
        self._times = list(times)
        self._i = 0

    def __call__(self):
        t = self._times[self._i]
        self._i = min(self._i + 1, len(self._times) - 1)
        return t

# ---- summarize / percentiles ----

def test_summarize_empty():
    s = summarize([])
    assert s.count == 0
    assert s.max_ms == 0.0

def test_summarize_basic_percentiles():
    s = summarize([float(x) for x in range(1, 101)])  # 1..100 ms
    assert s.count == 100
    assert s.min_ms == 1.0
    assert s.max_ms == 100.0
    assert abs(s.mean_ms - 50.5) < 1e-6
    assert s.p50_ms == 50.0
    assert s.p95_ms == 95.0
    assert s.p99_ms == 99.0

# ---- ResponseTimer with a fake clock ----

def test_timer_total_and_stages():
    # begin@0, mark capture@5ms, mark detect@20ms, mark emit@25ms, end@25ms
    clock = FakeClock([0.000, 0.005, 0.020, 0.025, 0.025])
    t = ResponseTimer(clock=clock)
    t.begin()
    assert abs(t.mark("capture") - 5.0) < 1e-6
    assert abs(t.mark("detect") - 15.0) < 1e-6
    assert abs(t.mark("emit") - 5.0) < 1e-6
    total = t.end()
    assert abs(total - 25.0) < 1e-6
    assert t.sample_count() == 1
    stages = t.stage_stats()
    assert abs(stages["detect"].mean_ms - 15.0) < 1e-6

def test_timer_multiple_frames_report():
    # frame1: begin@0 end@10ms (10ms); frame2: begin@10ms end@30ms (20ms)
    clock = FakeClock([0.0, 0.010, 0.010, 0.030, 0.030])
    t = ResponseTimer(clock=clock)
    t.begin(); t.end()
    t.begin(); t.end()
    rep = t.report()
    assert rep["total"]["count"] == 2
    assert rep["total"]["max_ms"] == 20.0
    assert rep["total"]["min_ms"] == 10.0

def test_timer_mark_without_begin_is_safe():
    t = ResponseTimer(clock=FakeClock([0.0, 0.001, 0.001]))
    # mark before begin should auto-begin and not crash
    t.mark("x")
    assert t.sample_count() == 0

# ---- ISO 13855 distance ----

def test_min_safety_distance_hand_speed():
    # Td=0.1s, Tm=0.2s, K=2000mm/s, C=0 -> S = 2000*0.3 = 600mm
    r = min_safety_distance(0.1, machine_time_s=0.2, approach_speed_mm_s=K_HAND_APPROACH)
    assert isinstance(r, SafetyDistanceResult)
    assert abs(r.total_time_s - 0.3) < 1e-9
    assert abs(r.min_distance_mm - 600.0) < 1e-6

def test_min_safety_distance_with_intrusion():
    # add C=100mm allowance
    r = min_safety_distance(0.1, machine_time_s=0.1, approach_speed_mm_s=1600.0, intrusion_mm=100.0)
    # S = 1600*0.2 + 100 = 420
    assert abs(r.min_distance_mm - 420.0) < 1e-6

def test_min_safety_distance_negative_times_clamped():
    r = min_safety_distance(-0.5, machine_time_s=-0.5)
    assert r.total_time_s == 0.0
    assert r.min_distance_mm == 0.0

def test_walking_constant_value():
    assert K_WALKING_APPROACH == 1600.0
    assert K_HAND_APPROACH == 2000.0

# ---- debounce delay ----

def test_debounce_time():
    # need 3 consecutive frames at 30fps -> (3-1)/30 = 0.0667s
    assert abs(debounce_time_s(3, 30.0) - (2 / 30.0)) < 1e-9

def test_debounce_single_frame_is_zero():
    assert debounce_time_s(1, 30.0) == 0.0
    assert debounce_time_s(5, 0.0) == 0.0

def test_debounce_feeds_into_distance():
    # realistic: single-frame proc 20ms + debounce 3 frames@30fps + machine 0.2s
    td = 0.020 + debounce_time_s(3, 30.0)
    r = min_safety_distance(td, machine_time_s=0.2)
    # td ~ 0.0867, total ~ 0.2867, S ~ 573mm
    assert 560 < r.min_distance_mm < 585
