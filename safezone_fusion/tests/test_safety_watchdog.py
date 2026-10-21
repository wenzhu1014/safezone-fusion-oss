"""Tests for the safety watchdog and producer heartbeat (2026-06-22)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.safety_watchdog import SafetyHeartbeat, SafetyWatchdog

def test_startup_grace_then_fail_safe_if_no_signal():
    wd = SafetyWatchdog(timeout_sec=0.5, startup_grace_sec=2.0)
    # within grace: no signal yet is tolerated
    s = wd.check(now_monotonic=100.0)      # first check sets t0
    assert s.fail_safe is False
    assert wd.check(now_monotonic=101.5).fail_safe is False
    # past grace, still nothing -> fail safe
    s = wd.check(now_monotonic=102.6)
    assert s.fail_safe is True
    assert s.reason == "no_signal_received"

def test_fresh_signal_is_ok():
    wd = SafetyWatchdog(timeout_sec=0.5)
    wd.record(seq=1, now_monotonic=10.0)
    s = wd.check(now_monotonic=10.2)
    assert s.fail_safe is False
    assert s.reason == "ok"
    assert s.last_seq == 1
    assert abs(s.age_sec - 0.2) < 1e-6

def test_stale_signal_triggers_fail_safe():
    wd = SafetyWatchdog(timeout_sec=0.5)
    wd.record(seq=5, now_monotonic=10.0)
    s = wd.check(now_monotonic=10.6)   # 0.6s > 0.5s timeout
    assert s.fail_safe is True
    assert s.reason == "signal_stale"
    assert s.last_seq == 5
    assert s.age_sec > 0.5

def test_recovery_after_new_signal():
    wd = SafetyWatchdog(timeout_sec=0.5)
    wd.record(seq=1, now_monotonic=10.0)
    assert wd.check(now_monotonic=10.6).fail_safe is True   # went stale
    wd.record(seq=2, now_monotonic=10.7)                    # new signal arrives
    s = wd.check(now_monotonic=10.8)
    assert s.fail_safe is False
    assert s.last_seq == 2

def test_out_of_order_seq_does_not_advance_but_newer_does():
    wd = SafetyWatchdog(timeout_sec=0.5)
    wd.record(seq=10, now_monotonic=10.0)
    wd.record(seq=7, now_monotonic=10.3)    # stale/duplicate older seq: ignored for freshness
    # freshness timer should still be anchored at the seq=10 receipt (10.0)
    s = wd.check(now_monotonic=10.6)
    assert s.fail_safe is True              # 0.6s since the seq=10 receipt
    assert s.last_seq == 10
    # a strictly newer seq refreshes
    wd.record(seq=11, now_monotonic=10.65)
    assert wd.check(now_monotonic=10.7).fail_safe is False

def test_invalid_timeout_rejected():
    try:
        SafetyWatchdog(timeout_sec=0)
        assert False
    except ValueError:
        pass

# --- producer heartbeat ---

def test_heartbeat_emits_first_call():
    hb = SafetyHeartbeat(min_interval_sec=0.2)
    assert hb.should_emit(now_monotonic=1.0, decision_key=("IDLE", "")) is True

def test_heartbeat_suppresses_until_interval_when_unchanged():
    hb = SafetyHeartbeat(min_interval_sec=0.2)
    hb.should_emit(1.0, ("IDLE", ""))
    # same decision, before interval -> suppressed
    assert hb.should_emit(1.1, ("IDLE", "")) is False
    # interval clearly elapsed -> emit (heartbeat keeps link alive)
    assert hb.should_emit(1.25, ("IDLE", "")) is True

def test_heartbeat_emits_immediately_on_change():
    hb = SafetyHeartbeat(min_interval_sec=0.2)
    hb.should_emit(1.0, ("IDLE", ""))
    # decision changed before interval -> emit right away (safety-relevant change)
    assert hb.should_emit(1.05, ("STOP", "human_in_stop_zone")) is True
