"""Tests for the product HUD overlay (2026-06-22)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.hud import (
    STATUS_ALLOW,
    STATUS_IDLE,
    STATUS_STOP,
    STATUS_WARNING,
    HudState,
    decision_to_status,
    reason_cn,
    render_hud,
)

def test_decision_to_status_stop_wins():
    assert decision_to_status("STOP", "STOP") == STATUS_STOP
    assert decision_to_status("IDLE", "STOP") == STATUS_STOP   # motor STOP forces stop
    assert decision_to_status("ALLOW_CART", "STOP") == STATUS_STOP

def test_decision_to_status_allow_and_warning_and_idle():
    assert decision_to_status("ALLOW_CART", "ALLOW_CART_DOCKING") == STATUS_ALLOW
    # a bare WARNING (no reason) is treated as proximity warning
    assert decision_to_status("WARNING", "HOLD") == STATUS_WARNING
    assert decision_to_status("IDLE", "HOLD") == STATUS_IDLE
    assert decision_to_status(None, None) == STATUS_IDLE

def test_cart_verification_maps_to_scanning_not_warning():
    from safezone_fusion.src.hud import STATUS_SCANNING
    # WARNING decision but reason is cart stabilisation -> scanning, NOT amber warning
    assert decision_to_status("WARNING", "HOLD", "cart_not_stable:still_time=0.6s<0.8s") == STATUS_SCANNING
    # IDLE with marker-not-stable / not-verified -> scanning
    assert decision_to_status("IDLE", "HOLD", "cart_not_verified:marker_not_stable:2/5") == STATUS_SCANNING
    assert decision_to_status("IDLE", "HOLD", "cart_not_verified:marker_not_visible") == STATUS_SCANNING
    # genuine proximity warning still maps to amber
    assert decision_to_status("WARNING", "HOLD", "object_in_warning_zone") == STATUS_WARNING

def test_reason_cn_known_and_fallback():
    assert reason_cn("human_in_stop_zone") == "检测到人体进入安全区"
    assert reason_cn("limb_in_stop_zone") == "检测到肢体进入安全区"
    # unknown reason falls back to raw
    assert reason_cn("some_new_reason") == "some_new_reason"

def test_render_hud_returns_same_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = HudState(status=STATUS_STOP, reason="human_in_stop_zone",
                     safety_zone_px=(200, 200, 440, 400), blink_on=True)
    out = render_hud(frame, state)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # something was drawn (banner/zone) -> frame changed
    assert not np.array_equal(out, frame)

def test_render_hud_fail_safe():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = HudState(status=STATUS_STOP, reason="", fail_safe=True)
    out = render_hud(frame, state)
    assert out.shape == frame.shape

def test_render_hud_handles_none_frame():
    assert render_hud(None, HudState(status=STATUS_IDLE)) is None

def test_render_hud_blink_off_dims_banner():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    on = render_hud(frame, HudState(status=STATUS_STOP, blink_on=True))
    off = render_hud(frame, HudState(status=STATUS_STOP, blink_on=False))
    # blink-on banner should be brighter (larger sum) than blink-off
    assert on.sum() >= off.sum()

def test_render_hud_zone_clamped_to_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # zone partly outside frame must not crash and stays valid
    state = HudState(status=STATUS_ALLOW, safety_zone_px=(-50, -50, 9999, 9999))
    out = render_hud(frame, state)
    assert out.shape == frame.shape
