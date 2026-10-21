from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import safezone_fusion.src.waiting_window as waiting_window_module
from safezone_fusion.src.models import MarkerObservation, Pose
from safezone_fusion.src.waiting_window import WaitingWindowChecker

def marker(
    *,
    stable_frames: int = 5,
    center=(320.0, 240.0),
    center_speed: float = 0.0,
    center_jitter: float = 0.0,
) -> MarkerObservation:
    return MarkerObservation(
        marker_id=123,
        pose=Pose(x=0.0, y=0.5, z=0.5),
        stable_frames=stable_frames,
        center_px=center,
        center_speed_px_s=center_speed,
        center_jitter_px=center_jitter,
        visible=True,
    )

def config() -> dict:
    return {
        "waiting_window": {
            "x_min": -0.35,
            "x_max": 0.35,
            "distance_min": 0.25,
            "distance_max": 0.70,
            "yaw_abs_max_deg": 10.0,
            "stability_mode": "angled_marker",
            "stable_marker_frames": 5,
            "stable_time_sec": 0.0,
            "angled_anchor_drift_max_px": 48.0,
            "angled_still_center_speed_max_px_s": 5.0,
            "angled_still_time_sec": 0.0,
            "wind_sway_anchor_drift_max_px": 8.0,
            "wind_sway_jitter_max_px": 3.0,
            "wind_sway_min_anchor_age_sec": 0.0,
        }
    }

def anchored_config() -> dict:
    cfg = config()
    cfg["waiting_window"].update(
        {
            "stability_mode": "anchored_position",
            "position_hold_time_sec": 1.0,
            "position_anchor_drift_max_px": 8.0,
            "position_history_radius_max_px": 3.0,
            "position_center_speed_warn_px_s": 5.0,
        }
    )
    return cfg

def test_angled_marker_accepts_still_marker():
    checker = WaitingWindowChecker(config())
    result = checker.check(marker(center_speed=2.0))
    assert result.in_window is True
    assert result.stable is True
    assert "cart_in_window_and_stable:angled_marker" in result.reason

def test_angled_marker_rejects_slow_creeping_marker_by_center_speed():
    checker = WaitingWindowChecker(config())
    first = checker.check(marker(center=(320.0, 240.0), center_speed=0.0))
    assert first.stable is True

    result = checker.check(marker(center=(330.0, 240.0), center_speed=6.0))
    assert result.in_window is True
    assert result.stable is False
    assert "center_speed=6.0px/s>5.0px/s" in result.reason

def test_angled_marker_rejects_large_anchor_drift():
    checker = WaitingWindowChecker(config())
    first = checker.check(marker(center=(320.0, 240.0), center_speed=0.0))
    assert first.stable is True
    result = checker.check(marker(center=(380.0, 240.0), center_speed=0.0))
    assert result.in_window is True
    assert result.stable is False
    assert "large_drift=60.0px>48.0px" in result.reason

def test_angled_marker_accepts_bounded_wind_sway():
    checker = WaitingWindowChecker(config())
    first = checker.check(marker(center=(320.0, 240.0), center_speed=0.0))
    assert first.stable is True

    result = checker.check(marker(center=(326.0, 240.0), center_speed=9.0, center_jitter=2.5))
    assert result.in_window is True
    assert result.stable is True
    assert "sway=yes" in result.reason

def test_angled_marker_rejects_fast_motion_outside_wind_sway_bounds():
    checker = WaitingWindowChecker(config())
    first = checker.check(marker(center=(320.0, 240.0), center_speed=0.0))
    assert first.stable is True

    result = checker.check(marker(center=(330.0, 240.0), center_speed=9.0, center_jitter=2.5))
    assert result.in_window is True
    assert result.stable is False
    assert "center_speed=9.0px/s>5.0px/s" in result.reason

def test_anchored_position_accepts_marker_held_inside_anchor(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    checker = WaitingWindowChecker(anchored_config())

    first = checker.check(marker(center=(320.0, 240.0), center_speed=0.0, center_jitter=0.5))
    assert first.stable is False
    assert "position_hold_time" in first.reason

    now[0] = 101.2
    result = checker.check(marker(center=(324.0, 240.0), center_speed=9.0, center_jitter=2.5))
    assert result.in_window is True
    assert result.stable is True
    assert "cart_in_window_and_stable:anchored_position" in result.reason
    assert "note=fast_local_sway" in result.reason

def test_anchored_position_rejects_marker_leaving_anchor(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    checker = WaitingWindowChecker(anchored_config())
    checker.check(marker(center=(320.0, 240.0), center_speed=0.0, center_jitter=0.5))

    now[0] = 101.2
    result = checker.check(marker(center=(330.0, 240.0), center_speed=6.0, center_jitter=2.5))
    assert result.in_window is True
    assert result.stable is False
    assert "position_anchor_drift=10.0px>8.0px" in result.reason

def test_anchored_position_rejects_marker_with_large_recent_radius(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    checker = WaitingWindowChecker(anchored_config())
    checker.check(marker(center=(320.0, 240.0), center_speed=0.0, center_jitter=0.5))

    now[0] = 101.2
    result = checker.check(marker(center=(324.0, 240.0), center_speed=4.0, center_jitter=3.5))
    assert result.in_window is True
    assert result.stable is False
    assert "position_radius=3.5px>3.0px" in result.reason
