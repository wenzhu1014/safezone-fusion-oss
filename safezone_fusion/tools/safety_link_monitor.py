"""Safety link monitor — reference consumer-side watchdog.

Simulates what the Android main board / PLC side should do: read the latest
SafeZone safety signal and FAIL-SAFE (force STOP) the moment signals go stale
or the link dies. Use it on-site as a link monitor, or as a working template
for the Android implementation.

It polls a LatestJsonSink file (written by the camera board) and applies
SafetyWatchdog. If the file stops updating (board crash, cable pulled, process
killed), the watchdog flips to fail-safe.

Usage:
    python tools/safety_link_monitor.py --latest out/now.json --timeout 0.5

Exit codes are not used for control; this is a monitor/demonstrator. A real PLC
integration would drive a stop output instead of printing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.safety_watchdog import SafetyWatchdog

def _read_latest(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("seq")), bool(data.get("stop")), data
    except Exception:
        return None, None, None

def main() -> None:
    ap = argparse.ArgumentParser(description="SafeZone consumer-side safety link monitor")
    ap.add_argument("--latest", required=True, help="path to LatestJsonSink file (e.g. out/now.json)")
    ap.add_argument("--timeout", type=float, default=0.5, help="stale timeout seconds -> fail-safe")
    ap.add_argument("--startup-grace", type=float, default=3.0, help="startup grace seconds")
    ap.add_argument("--poll-ms", type=int, default=50, help="poll interval ms")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after N seconds (0 = run forever)")
    args = ap.parse_args()

    latest = Path(args.latest)
    wd = SafetyWatchdog(timeout_sec=args.timeout, startup_grace_sec=args.startup_grace)

    print(f"[monitor] watching {latest} | timeout={args.timeout}s grace={args.startup_grace}s")
    print("[monitor] FAIL-SAFE = consumer must force STOP (link stale/dead)")

    last_printed = None
    start = time.monotonic()
    while True:
        now = time.monotonic()
        seq, stop, _ = _read_latest(latest)
        if seq is not None:
            wd.record(seq, now)
        state = wd.check(now)

        # effective command: STOP if signal says stop OR watchdog fails safe
        effective_stop = bool(stop) or state.fail_safe
        status = (
            f"fail_safe={state.fail_safe} reason={state.reason} "
            f"seq={state.last_seq} age={None if state.age_sec is None else round(state.age_sec,3)} "
            f"signal_stop={stop} => EFFECTIVE_STOP={effective_stop}"
        )
        if status != last_printed:
            print(f"[monitor] {status}")
            last_printed = status

        if args.max_seconds and (now - start) >= args.max_seconds:
            break
        time.sleep(args.poll_ms / 1000.0)

if __name__ == "__main__":
    main()
