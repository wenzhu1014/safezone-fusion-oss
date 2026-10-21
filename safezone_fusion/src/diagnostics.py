from __future__ import annotations

"""Per-frame diagnostic recorder for on-site debugging (2026-06-22).

When something misbehaves on a real cabinet ("it stopped and I don't know why",
"it didn't stop when I expected"), the operator needs to see exactly what each
layer decided, frame by frame, and be able to replay it later. This recorder
writes one JSON line per frame capturing the full decision trace:

  * timestamp, frame index, measured fps
  * marker: visible / coasting / authorized / center / stable_frames
  * each object observation (label, position) the detectors emitted
  * lens health (blinded?), operating state (ready/degraded)
  * final decision + reason + stop boolean + safety seq

It is intentionally dependency-free (stdlib json only) and append-only, so it is
safe to leave running on site. A companion summary tool turns the log into a
human-readable "why did it stop" timeline.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

@dataclass
class FrameDiag:
    """One frame's worth of cross-layer diagnostic state."""

    frame: int
    ts: float
    fps: float = 0.0
    # marker
    marker_visible: Optional[bool] = None
    marker_coasting: Optional[bool] = None
    marker_authorized: Optional[bool] = None
    marker_center: Optional[tuple] = None
    marker_stable_frames: Optional[int] = None
    # detectors
    objects: list = field(default_factory=list)   # [{"label":..., "x":..., "z":...}]
    # health / state gates
    lens_blinded: Optional[bool] = None
    lens_reason: str = ""
    op_ready: Optional[bool] = None
    op_degraded: Optional[bool] = None
    op_reason: str = ""
    # final
    decision: str = ""
    reason: str = ""
    stop: Optional[bool] = None
    seq: Optional[int] = None
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

class DiagnosticRecorder:
    """Append-only JSONL recorder. One line per frame.

    Usage::

        rec = DiagnosticRecorder("out/diag.jsonl")   # or None to disable
        d = FrameDiag(frame=i, ts=time.time())
        ... fill d ...
        rec.write(d)
        rec.close()
    """

    def __init__(self, path: Optional[str], flush_every: int = 1):
        self.path = path
        self._fh = None
        self._n = 0
        self._flush_every = max(1, flush_every)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(p, "a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def write(self, diag: FrameDiag) -> None:
        if self._fh is None:
            return
        self._fh.write(diag.to_json() + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None

# ---- offline analysis helpers (used by tools/diag_summary.py) ----

def load_diag(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def stop_transitions(rows: list) -> list:
    """Return frames where the stop boolean flips, with the reason -- the
    "why did it stop / start" timeline an operator actually wants."""
    out = []
    prev = None
    for r in rows:
        s = r.get("stop")
        if s != prev:
            out.append({
                "frame": r.get("frame"),
                "ts": r.get("ts"),
                "stop": s,
                "reason": r.get("reason") or r.get("op_reason") or r.get("lens_reason"),
                "decision": r.get("decision"),
            })
            prev = s
    return out

def summarize(rows: list) -> dict:
    """High-level counts for a quick health read of a recording."""
    if not rows:
        return {"frames": 0}
    stops = sum(1 for r in rows if r.get("stop"))
    reasons: dict = {}
    for r in rows:
        if r.get("stop"):
            key = r.get("reason") or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
    fps_vals = [r.get("fps", 0.0) for r in rows if r.get("fps")]
    return {
        "frames": len(rows),
        "stop_frames": stops,
        "stop_ratio": round(stops / len(rows), 3),
        "stop_reasons": reasons,
        "fps_mean": round(sum(fps_vals) / len(fps_vals), 1) if fps_vals else 0.0,
        "fps_min": round(min(fps_vals), 1) if fps_vals else 0.0,
        "transitions": len(stop_transitions(rows)),
    }
