from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.diagnostics import (
    DiagnosticRecorder,
    FrameDiag,
    load_diag,
    stop_transitions,
    summarize,
)

def test_recorder_writes_jsonl(tmp_path):
    p = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(str(p))
    assert rec.enabled
    for i in range(3):
        rec.write(FrameDiag(frame=i, ts=100.0 + i, stop=(i == 1), reason="limb_in_stop_zone" if i == 1 else ""))
    rec.close()
    rows = load_diag(str(p))
    assert len(rows) == 3
    assert rows[1]["stop"] is True
    assert rows[1]["reason"] == "limb_in_stop_zone"

def test_disabled_recorder_is_noop():
    rec = DiagnosticRecorder(None)
    assert rec.enabled is False
    rec.write(FrameDiag(frame=0, ts=0.0))  # must not raise
    rec.close()

def test_stop_transitions_detects_flips():
    rows = [
        {"frame": 0, "ts": 0.0, "stop": False},
        {"frame": 1, "ts": 0.1, "stop": False},
        {"frame": 2, "ts": 0.2, "stop": True, "reason": "limb_in_stop_zone"},
        {"frame": 3, "ts": 0.3, "stop": True},
        {"frame": 4, "ts": 0.4, "stop": False},
    ]
    trans = stop_transitions(rows)
    # False(initial) -> True -> False  == 3 transition points
    assert [t["stop"] for t in trans] == [False, True, False]
    assert trans[1]["reason"] == "limb_in_stop_zone"
    assert trans[1]["frame"] == 2

def test_summarize_counts_and_reasons():
    rows = [
        {"frame": 0, "ts": 0.0, "stop": False, "fps": 30.0},
        {"frame": 1, "ts": 0.1, "stop": True, "reason": "limb_in_stop_zone", "fps": 29.0},
        {"frame": 2, "ts": 0.2, "stop": True, "reason": "limb_in_stop_zone", "fps": 28.0},
        {"frame": 3, "ts": 0.3, "stop": True, "reason": "lens_blinded", "fps": 10.0},
    ]
    s = summarize(rows)
    assert s["frames"] == 4
    assert s["stop_frames"] == 3
    assert s["stop_reasons"]["limb_in_stop_zone"] == 2
    assert s["stop_reasons"]["lens_blinded"] == 1
    assert s["fps_min"] == 10.0

def test_summarize_empty():
    assert summarize([])["frames"] == 0

def test_framediag_roundtrip():
    d = FrameDiag(frame=5, ts=12.5, decision="STOP", reason="limb_in_stop_zone",
                  stop=True, objects=[{"label": "hand", "x": 0.0, "z": 0.4}])
    import json
    parsed = json.loads(d.to_json())
    assert parsed["frame"] == 5
    assert parsed["objects"][0]["label"] == "hand"
