from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.operating_state import (
    OperatingStateConfig,
    OperatingStateMonitor,
)

class FakeClock:
    """Advance by a controllable dt each call."""

    def __init__(self, dt=1 / 30.0, start=1000.0):
        self.t = start
        self.dt = dt

    def __call__(self):
        v = self.t
        self.t += self.dt
        return v

def test_startup_requires_stop_until_warmed():
    cfg = OperatingStateConfig(warmup_frames=10, warmup_min_seconds=0.2)
    clk = FakeClock(dt=1 / 30.0)
    mon = OperatingStateMonitor(cfg, clock=clk)
    # first frames: not ready -> stop_required True
    for i in range(9):
        r = mon.tick()
        assert r.ready is False
        assert r.stop_required is True
        assert "warming_up" in r.reason
    # 10th frame + enough time -> ready
    r = mon.tick()
    assert r.ready is True
    assert r.stop_required is False

def test_warmup_needs_both_frames_and_time():
    # plenty of frames but arriving too fast (not enough wall time)
    cfg = OperatingStateConfig(warmup_frames=5, warmup_min_seconds=1.0)
    clk = FakeClock(dt=0.01)  # 100 fps -> 5 frames in 0.05s < 1.0s
    mon = OperatingStateMonitor(cfg, clock=clk)
    for _ in range(5):
        r = mon.tick()
    assert r.ready is False  # frames met but time not met
    assert r.stop_required is True

def test_healthy_fps_not_degraded():
    cfg = OperatingStateConfig(warmup_frames=3, warmup_min_seconds=0.0,
                               min_fps=12, recover_fps=15, degrade_after=5)
    clk = FakeClock(dt=1 / 30.0)  # 30 fps
    mon = OperatingStateMonitor(cfg, clock=clk)
    r = None
    for _ in range(40):
        r = mon.tick()
    assert r.ready is True
    assert r.degraded is False
    assert r.fps == pytest.approx(30.0, abs=2.0)

def test_low_fps_degrades_after_streak():
    cfg = OperatingStateConfig(warmup_frames=3, warmup_min_seconds=0.0,
                               min_fps=12, degrade_after=5, fps_window=10)
    clk = FakeClock(dt=1 / 5.0)  # 5 fps -> below floor
    mon = OperatingStateMonitor(cfg, clock=clk)
    degraded_seen = False
    for _ in range(30):
        r = mon.tick()
        degraded_seen = degraded_seen or r.degraded
    assert degraded_seen is True
    assert r.stop_required is True
    assert "fps_degraded" in r.reason

def test_degraded_recovers_with_hysteresis():
    cfg = OperatingStateConfig(warmup_frames=3, warmup_min_seconds=0.0,
                               min_fps=12, recover_fps=15,
                               degrade_after=3, recover_after=5, fps_window=8)
    mon = OperatingStateMonitor(cfg)
    # manual clock control: warm up fast first
    import itertools
    # warmup at 30fps
    mon._clock = FakeClock(dt=1 / 30.0)
    for _ in range(5):
        mon.tick()
    # drop to 5 fps to degrade
    mon._clock = FakeClock(dt=1 / 5.0, start=mon._stamps[-1] + 0.2)
    for _ in range(10):
        r = mon.tick()
    assert r.degraded is True
    # back to 30 fps to recover
    mon._clock = FakeClock(dt=1 / 30.0, start=mon._stamps[-1] + 0.03)
    for _ in range(15):
        r = mon.tick()
    assert r.degraded is False

def test_disabled_monitor_always_ready():
    cfg = OperatingStateConfig(enabled=False)
    mon = OperatingStateMonitor(cfg)
    r = mon.tick()
    assert r.ready is True
    assert r.stop_required is False

def test_reset_returns_to_warmup():
    cfg = OperatingStateConfig(warmup_frames=3, warmup_min_seconds=0.0)
    mon = OperatingStateMonitor(cfg)
    for _ in range(5):
        mon.tick()
    assert mon.tick().ready is True
    mon.reset()
    r = mon.tick()
    assert r.ready is False
    assert r.stop_required is True
