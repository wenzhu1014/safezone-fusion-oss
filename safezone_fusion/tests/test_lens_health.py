from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.lens_health import (
    LensFault,
    LensHealthConfig,
    LensHealthMonitor,
    analyze_frame,
    apply_lens_failsafe,
)
from safezone_fusion.src.interfaces import ActuatorOutput, MotorAction

def _normal_scene(w=160, h=120, seed=0):
    """A textured, well-exposed frame: passes the health check."""
    rng = np.random.default_rng(seed)
    return (rng.integers(30, 220, size=(h, w, 3))).astype("uint8")

def _black(w=160, h=120):
    return np.zeros((h, w, 3), dtype="uint8")

def _white(w=160, h=120):
    return np.full((h, w, 3), 255, dtype="uint8")

def _flat_grey(w=160, h=120, value=128):
    return np.full((h, w, 3), value, dtype="uint8")

# ---- stateless single-frame analysis ----

def test_normal_scene_is_healthy():
    cfg = LensHealthConfig()
    bad, fault, metrics = analyze_frame(_normal_scene(), cfg)
    assert bad is False
    assert fault is LensFault.NONE
    assert metrics["detail"] > cfg.flat_detail_max

def test_black_frame_is_covered_dark():
    cfg = LensHealthConfig()
    bad, fault, _ = analyze_frame(_black(), cfg)
    assert bad is True
    assert fault is LensFault.COVERED_DARK

def test_white_frame_is_glare_saturated():
    cfg = LensHealthConfig()
    bad, fault, _ = analyze_frame(_white(), cfg)
    assert bad is True
    assert fault is LensFault.GLARE_SATURATED

def test_flat_grey_frame_is_flat_featureless():
    cfg = LensHealthConfig()
    bad, fault, _ = analyze_frame(_flat_grey(value=128), cfg)
    assert bad is True
    assert fault is LensFault.FLAT_FEATURELESS

def test_none_frame_is_bad_fail_safe():
    cfg = LensHealthConfig()
    bad, fault, _ = analyze_frame(None, cfg)
    assert bad is True
    assert fault is LensFault.COVERED_DARK

def test_mono_2d_frame_supported():
    cfg = LensHealthConfig()
    gray = _normal_scene()[:, :, 0]
    bad, fault, _ = analyze_frame(gray, cfg)
    assert bad is False

# ---- stateful hysteresis ----

def test_latch_requires_consecutive_bad_frames():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 3, "release_frames": 8}})
    # two bad frames: not latched yet
    assert mon.update(_black()).blinded is False
    assert mon.update(_black()).blinded is False
    # third consecutive bad frame: latched
    r = mon.update(_black())
    assert r.blinded is True
    assert r.fault is LensFault.COVERED_DARK

def test_single_bad_frame_does_not_latch():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 3}})
    assert mon.update(_black()).blinded is False
    # one good frame resets the bad streak
    assert mon.update(_normal_scene()).blinded is False
    assert mon.update(_black()).blinded is False
    assert mon.update(_black()).blinded is False
    # still not latched because the streak was broken
    assert mon.update(_normal_scene()).blinded is False

def test_release_is_slower_than_latch():
    cfg = {"lens_health": {"enter_frames": 2, "release_frames": 5}}
    mon = LensHealthMonitor(cfg)
    mon.update(_black())
    assert mon.update(_black()).blinded is True  # latched
    # a few good frames, but fewer than release_frames -> still blinded
    for _ in range(4):
        assert mon.update(_normal_scene()).blinded is True
    # fifth good frame clears it
    assert mon.update(_normal_scene()).blinded is False

def test_bad_frame_during_recovery_resets_release():
    cfg = {"lens_health": {"enter_frames": 2, "release_frames": 4}}
    mon = LensHealthMonitor(cfg)
    mon.update(_white())
    assert mon.update(_white()).blinded is True
    for _ in range(3):
        mon.update(_normal_scene())  # almost recovered
    # a bad frame interrupts recovery
    assert mon.update(_white()).blinded is True
    # now needs the full release count again
    for _ in range(3):
        assert mon.update(_normal_scene()).blinded is True
    assert mon.update(_normal_scene()).blinded is False

def test_disabled_monitor_never_blinds():
    mon = LensHealthMonitor({"lens_health": {"enabled": False}})
    for _ in range(10):
        assert mon.update(_black()).blinded is False

def test_reset_clears_latch():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 1}})
    assert mon.update(_black()).blinded is True
    mon.reset()
    assert mon.blinded is False

def test_reason_strings_are_informative():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 2}})
    r1 = mon.update(_black())
    assert "lens_degrading:covered_dark:1/2" == r1.reason
    r2 = mon.update(_black())
    assert r2.reason == "lens_blinded:covered_dark"
    # recovery
    mon2 = LensHealthMonitor({"lens_health": {"enter_frames": 1, "release_frames": 1}})
    mon2.update(_black())
    ok = mon2.update(_normal_scene())
    assert ok.reason == "lens_ok"

# ---- fail-safe override into the actuator ----

def test_failsafe_overrides_actuator_to_stop():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 1}})
    lens = mon.update(_black())
    assert lens.blinded is True
    act = ActuatorOutput(motor=MotorAction.ALLOW_CART_DOCKING, reason="authorized_cart")
    apply_lens_failsafe(act, lens)
    assert act.motor is MotorAction.STOP
    assert act.reason == "lens_blinded:covered_dark"
    assert act.debug.get("lens_failsafe") is True
    assert act.debug.get("lens_fault") == "covered_dark"

def test_failsafe_leaves_actuator_untouched_when_healthy():
    mon = LensHealthMonitor({"lens_health": {"enter_frames": 2}})
    lens = mon.update(_normal_scene())
    assert lens.blinded is False
    act = ActuatorOutput(motor=MotorAction.ALLOW_CART_DOCKING, reason="authorized_cart")
    apply_lens_failsafe(act, lens)
    assert act.motor is MotorAction.ALLOW_CART_DOCKING
    assert act.reason == "authorized_cart"
    assert "lens_failsafe" not in act.debug

def test_failsafe_handles_none_lens():
    act = ActuatorOutput(motor=MotorAction.HOLD)
    out = apply_lens_failsafe(act, None)
    assert out.motor is MotorAction.HOLD
