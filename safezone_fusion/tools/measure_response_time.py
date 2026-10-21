from __future__ import annotations

"""Measure SafeZone camera-side end-to-end response time and derive the
ISO 13855 minimum safety distance.

What it measures (T_detect)
---------------------------
The camera board owns the time from "a hazard is present in the captured frame"
to "stop=true is published toward the PLC". That is:

    T_detect = single-frame processing latency (capture->detect->decide->emit)
             + debounce latency ((min_consecutive - 1) / fps)

Single-frame latency is measured by running the *real* vision + fusion pipeline
on synthetic frames many times and taking the worst case (p95/p99/max). The
debounce term comes from how many consecutive occupied frames we require before
latching STOP (roi_motion ``min_consecutive`` / lens ``enter_frames``).

Then we plug the worst-case T_detect into ISO 13855:

    S = K * (T_detect + T_machine) + C

T_machine (PLC reaction + servo/mechanical stop) is supplied by the integrator
via --machine-ms; we print the resulting safety distance for a few values so
the PLC team can read off their number.

Usage
-----
    python -X utf8 tools/measure_response_time.py --frames 500 --fps 30 \
        --min-consecutive 3 --machine-ms 100 200 300

This needs no physical camera (uses synthetic frames), so it is reproducible on
any machine and in CI. Numbers will differ on the real camera board CPU; rerun
there for the deployment record.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import BusinessInput, OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.response_time import (
    K_HAND_APPROACH,
    K_WALKING_APPROACH,
    ResponseTimer,
    debounce_time_s,
    min_safety_distance,
)
from safezone_fusion.src.vision.interfaces import CameraPacket, PirPacket, VisionSensorFusion
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.lens_health import LensHealthMonitor, apply_lens_failsafe
from safezone_fusion.src.vision.object_detector_factory import resolve_object_detectors

class ConstantPirReader:
    def __init__(self, triggered: bool = False):
        self.triggered = triggered

    def read(self) -> PirPacket:
        return PirPacket(triggered=self.triggered)

def make_hazard_frame(w: int, h: int, rng) -> np.ndarray:
    """A textured frame with a bright 'intruding object' blob in the ROI."""
    frame = rng.integers(40, 90, size=(h, w, 3), dtype=np.uint8)
    # a person/limb-like blob low-center (typical danger-zone ROI)
    y0, y1 = int(h * 0.62), int(h * 0.95)
    x0, x1 = int(w * 0.35), int(w * 0.65)
    frame[y0:y1, x0:x1] = rng.integers(150, 220, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500, help="number of frames to time")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0, help="capture FPS (for debounce delay)")
    ap.add_argument("--min-consecutive", type=int, default=3,
                    help="consecutive occupied frames required before STOP latches (debounce)")
    ap.add_argument("--detectors", type=str, default="roi_motion",
                    help="object detectors, comma-separated (e.g. 'mediapipe_hand,roi_motion')")
    ap.add_argument("--machine-ms", type=float, nargs="+", default=[100.0, 200.0, 300.0],
                    help="candidate PLC+servo stopping times (ms) to tabulate")
    ap.add_argument("--approach", choices=["hand", "walking"], default="hand")
    ap.add_argument("--intrusion-mm", type=float, default=0.0, help="ISO 13855 C term")
    args = ap.parse_args()

    config = load_config(ROOT / "safezone_fusion" / "config.yaml")
    orchestrator = SafeZoneOrchestrator(config)
    detector = OpenCVArucoMarkerDetector(dictionary_name="DICT_6X6_250", marker_size_m=0.10,
                                          target_marker_ids=None)
    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    _, detectors = resolve_object_detectors(config, names)
    vision = VisionSensorFusion(marker_detector=detector, object_detector=detectors,
                                pir_reader=ConstantPirReader(False))
    lens_monitor = LensHealthMonitor(config)
    business = BusinessInput(mode=OperationMode.CART_DOCKING, docking_requested=True)

    timer = ResponseTimer()
    rng = np.random.default_rng(7)

    print(f"SafeZone response-time benchmark  ({args.width}x{args.height}, detectors={names})")
    print(f"warming up pipeline ...")
    # warmup: motion/occupancy detectors need a few frames to build background
    for _ in range(30):
        f = make_hazard_frame(args.width, args.height, rng)
        pkt = CameraPacket(rgb=f, timestamp=time.time())
        s = vision.read(pkt)
        inp = RuntimeInput(business=business, sensors=s, tick_id=0)
        out = orchestrator.tick(inp)
        lens_monitor.update(f)

    print(f"measuring {args.frames} frames ...")
    for i in range(args.frames):
        frame = make_hazard_frame(args.width, args.height, rng)
        timer.begin()
        pkt = CameraPacket(rgb=frame, timestamp=time.time())
        timer.mark("capture")
        sensors = vision.read(pkt)
        timer.mark("detect")
        inp = RuntimeInput(business=business, sensors=sensors, tick_id=i)
        out = orchestrator.tick(inp)
        timer.mark("decide")
        lens = lens_monitor.update(frame)
        if lens.blinded:
            apply_lens_failsafe(out.actuator, lens)
        timer.mark("emit")
        timer.end()

    rep = timer.report()
    total = rep["total"]
    print("\n=== Single-frame processing latency (ms) ===")
    print(f"  count={total['count']}  mean={total['mean_ms']}  p50={total['p50_ms']}  "
          f"p95={total['p95_ms']}  p99={total['p99_ms']}  max={total['max_ms']}")
    print("  per-stage mean (ms):")
    for stage, st in rep["stages"].items():
        print(f"    {stage:8s} mean={st['mean_ms']:7.2f}  p95={st['p95_ms']:7.2f}  max={st['max_ms']:7.2f}")

    # worst-case single-frame proc: use p99 (robust against a single OS hiccup max)
    proc_p99_s = total["p99_ms"] / 1000.0
    proc_max_s = total["max_ms"] / 1000.0
    debounce_s = debounce_time_s(args.min_consecutive, args.fps)
    td_typical = proc_p99_s + debounce_s
    td_worst = proc_max_s + debounce_s

    print("\n=== T_detect (camera-side response) ===")
    print(f"  single-frame p99 : {proc_p99_s*1000:.1f} ms")
    print(f"  debounce ({args.min_consecutive} frames @ {args.fps:g}fps): {debounce_s*1000:.1f} ms")
    print(f"  T_detect (p99 + debounce): {td_typical*1000:.1f} ms")
    print(f"  T_detect (max + debounce, worst): {td_worst*1000:.1f} ms")

    K = K_HAND_APPROACH if args.approach == "hand" else K_WALKING_APPROACH
    print(f"\n=== ISO 13855 minimum safety distance  S = K*(Td+Tm) + C ===")
    print(f"  K (approach speed) = {K:g} mm/s ({args.approach})   C (intrusion) = {args.intrusion_mm:g} mm")
    print(f"  Td (worst camera-side) = {td_worst*1000:.1f} ms")
    print(f"  {'T_machine(ms)':>14} | {'S_typical(mm)':>14} | {'S_worst(mm)':>12}")
    print(f"  {'-'*14}-+-{'-'*14}-+-{'-'*12}")
    for tm_ms in args.machine_ms:
        tm_s = tm_ms / 1000.0
        s_typ = min_safety_distance(td_typical, machine_time_s=tm_s, approach_speed_mm_s=K,
                                    intrusion_mm=args.intrusion_mm).min_distance_mm
        s_wst = min_safety_distance(td_worst, machine_time_s=tm_s, approach_speed_mm_s=K,
                                    intrusion_mm=args.intrusion_mm).min_distance_mm
        print(f"  {tm_ms:>14.0f} | {s_typ:>14.1f} | {s_wst:>12.1f}")

    print("\nNote: Td measured on THIS machine. Re-run on the camera board CPU for the")
    print("deployment record. Hand T_detect(worst) to the PLC team; they add T_machine.")

if __name__ == "__main__":
    main()
