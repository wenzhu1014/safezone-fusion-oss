"""Tests for the pluggable safety-signal output layer (2026-06-22)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.interfaces import ActuatorOutput, DoorAction, MotorAction
from safezone_fusion.src.safety_signal import (
    SCHEMA,
    JsonlFileSink,
    LatestJsonSink,
    SafetySignalEmitter,
    build_signal,
    build_sinks_from_spec,
)

def _out(motor=MotorAction.HOLD, decision="IDLE", reason="", voice=None):
    return ActuatorOutput(motor=motor, door=DoorAction.NONE, voice=voice,
                          reason=reason, raw_decision=decision)

def test_stop_true_for_stop_motor():
    sig = build_signal(_out(MotorAction.STOP, "STOP", "human_in_stop_zone"), seq=1)
    assert sig["stop"] is True
    assert sig["schema"] == SCHEMA
    assert sig["motor"] == "STOP"
    assert sig["decision"] == "STOP"
    assert sig["reason"] == "human_in_stop_zone"

def test_stop_true_for_lock_still():
    sig = build_signal(_out(MotorAction.LOCK_STILL, "WARNING"), seq=2)
    assert sig["stop"] is True

def test_stop_false_for_allow():
    sig = build_signal(_out(MotorAction.ALLOW_CART_DOCKING, "ALLOW_CART"), seq=3)
    assert sig["stop"] is False

def test_stop_false_for_hold_idle():
    sig = build_signal(_out(MotorAction.HOLD, "IDLE"), seq=4)
    assert sig["stop"] is False

def test_seq_is_monotonic_and_fans_out(tmp_path):
    captured = []

    class _Capture:
        def emit(self, signal):
            captured.append(signal)
        def close(self):
            pass

    emitter = SafetySignalEmitter([_Capture(), _Capture()])
    s1 = emitter.emit(_out(MotorAction.STOP, "STOP"))
    s2 = emitter.emit(_out(MotorAction.HOLD, "IDLE"))
    assert s1["seq"] == 1 and s2["seq"] == 2
    # two sinks each received both signals
    assert len(captured) == 4
    assert [c["seq"] for c in captured] == [1, 1, 2, 2]

def test_failing_sink_does_not_break_others():
    errors = []

    class _Bad:
        def emit(self, signal):
            raise RuntimeError("transport down")
        def close(self):
            pass

    class _Good:
        def __init__(self):
            self.got = []
        def emit(self, signal):
            self.got.append(signal)
        def close(self):
            pass

    good = _Good()
    emitter = SafetySignalEmitter([_Bad(), good], on_error=lambda s, e: errors.append(e))
    emitter.emit(_out(MotorAction.STOP, "STOP"))
    assert len(good.got) == 1          # good sink still delivered
    assert len(errors) == 1            # error was reported, not raised

def test_jsonl_file_sink_writes_lines(tmp_path):
    path = tmp_path / "sig.jsonl"
    emitter = SafetySignalEmitter([JsonlFileSink(path)])
    emitter.emit(_out(MotorAction.STOP, "STOP", "human_in_stop_zone"))
    emitter.emit(_out(MotorAction.HOLD, "IDLE"))
    emitter.close()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["stop"] is True
    assert json.loads(lines[1])["stop"] is False

def test_latest_json_sink_overwrites(tmp_path):
    path = tmp_path / "now.json"
    emitter = SafetySignalEmitter([LatestJsonSink(path)])
    emitter.emit(_out(MotorAction.STOP, "STOP"))
    emitter.emit(_out(MotorAction.ALLOW_CART_DOCKING, "ALLOW_CART"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seq"] == 2          # only the latest remains
    assert data["stop"] is False

def test_build_sinks_from_spec(tmp_path):
    p = tmp_path / "out.jsonl"
    sinks = build_sinks_from_spec(f"stdout,file:{p}")
    assert len(sinks) == 2
    # empty spec -> no sinks
    assert build_sinks_from_spec("") == []
    assert build_sinks_from_spec(None) == []

def test_build_sinks_audit_always_appended(tmp_path):
    audit = tmp_path / "audit.jsonl"
    sinks = build_sinks_from_spec("stdout", audit_path=audit)
    assert len(sinks) == 2  # stdout + audit file

def test_build_sinks_rejects_unknown(tmp_path):
    try:
        build_sinks_from_spec("carrierpigeon:foo")
        assert False, "should have raised"
    except ValueError:
        pass
