from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.demo_live_camera import apply_environment_profile, build_roi_motion_config
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.interfaces import CameraPacket, PirPacket, VisionSensorFusion
from safezone_fusion.src.vision.mock_pipeline import MockObjectDetector  # re-exported for compat
from safezone_fusion.src.vision.object_detector_factory import build_object_detector, resolve_object_detector
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector  # re-exported for compat
from safezone_fusion.src.vision.skin_motion_detector import SkinMotionObjectDetector  # re-exported for compat
import safezone_fusion.src.waiting_window as waiting_window_module

class ConstantPirReader:
    def read(self):
        return PirPacket(False, timestamp=time.time())

CASES: list[dict[str, Any]] = [
    {
        "name": "static_marker_720",
        "video": "captures/clips/static_marker_20260617-103852_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_min": 0.99, "allow_min": 30, "stop_eq": 0},
    },
    {
        "name": "slow_move_marker_720",
        "video": "captures/clips/slow_move_marker_20260617-104247_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_min": 0.99, "warning_min": 40, "stop_eq": 0},
    },
    {
        "name": "sway_marker_720_indoor",
        "video": "captures/clips/sway_marker_20260617-105158_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_min": 0.99, "warning_min": 50, "stop_eq": 0},
    },
    {
        "name": "light_sway_marker_720_indoor",
        "video": "captures/clips/light_sway_marker_20260617-105345_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_min": 0.99, "allow_min": 30, "stop_eq": 0},
    },
    {
        "name": "marker_missing_720",
        "video": "captures/clips/marker_missing_20260617-105627_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_min": 0.60, "idle_min": 10, "allow_min": 5, "stop_eq": 0, "require_missing_then_recover": True},
    },
    {
        "name": "hand_stop_zone_roi",
        "video": "captures/clips/hand_stop_zone_20260617-112315_1280x720.mp4",
        "environment": "indoor",
        "roi_motion": True,
        "expect": {"detection_rate_min": 0.99, "allow_min": 30, "stop_min": 5, "require_allow_stop_allow": True},
    },
    {
        "name": "hand_stop_zone_skin_motion",
        "video": "captures/clips/hand_stop_zone_20260617-112315_1280x720.mp4",
        "environment": "indoor",
        "object_detector": "skin_motion",
        "expect": {"detection_rate_min": 0.99, "allow_min": 30, "stop_min": 5, "require_allow_stop_allow": True},
    },
    {
        "name": "non_hand_motion_clean_roi_motion_stops",
        "video": "captures/clips/non_hand_motion_20260617-132051_1280x720.mp4",
        "environment": "indoor",
        "object_detector": "roi_motion",
        "expect": {"detection_rate_min": 0.99, "stop_min": 1},
    },
    {
        "name": "non_hand_motion_clean_skin_motion_clean",
        "video": "captures/clips/non_hand_motion_20260617-132051_1280x720.mp4",
        "environment": "indoor",
        "object_detector": "skin_motion",
        "expect": {"detection_rate_min": 0.99, "allow_min": 30, "stop_eq": 0},
    },
    {
        "name": "non_hand_motion_aggressive_roi_motion_stops",
        "video": "captures/clips/non_hand_motion_20260617-132027_1280x720.mp4",
        "environment": "indoor",
        "object_detector": "roi_motion",
        "expect": {"stop_min": 30},
    },
    {
        "name": "non_hand_motion_aggressive_skin_motion_clean",
        "video": "captures/clips/non_hand_motion_20260617-132027_1280x720.mp4",
        "environment": "indoor",
        "object_detector": "skin_motion",
        "expect": {"stop_eq": 0},
    },
    {
        "name": "static_marker_480_negative",
        "video": "captures/clips/static_marker_480_20260617-104659_640x480.mp4",
        "environment": "indoor",
        "roi_motion": False,
        "expect": {"detection_rate_max": 0.05, "allow_eq": 0, "stop_eq": 0},
    },
]

def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row["decision"]] = out.get(row["decision"], 0) + 1
    return out

def _has_sequence(rows: list[dict[str, Any]], sequence: list[str]) -> bool:
    idx = 0
    for row in rows:
        if row["decision"] == sequence[idx]:
            idx += 1
            if idx == len(sequence):
                return True
    return False

def _missing_then_recover(rows: list[dict[str, Any]]) -> bool:
    seen_missing = False
    for row in rows:
        if not row["marker_detected"]:
            seen_missing = True
        if seen_missing and row["marker_detected"] and row["decision"] == "ALLOW_CART":
            return True
    return False

def replay_case(case: dict[str, Any], config: dict[str, Any], output_dir: Path, max_frames: int = 0) -> dict[str, Any]:
    video = ROOT / case["video"]
    if not video.exists():
        return {"name": case["name"], "video": str(video), "pass": False, "errors": ["video_not_found"]}

    cfg = json.loads(json.dumps(config))
    apply_environment_profile(cfg, case.get("environment", "indoor"))
    orchestrator = SafeZoneOrchestrator(cfg)
    detector = OpenCVArucoMarkerDetector(
        dictionary_name="DICT_6X6_250",
        marker_size_m=0.10,
        target_marker_ids={123},
    )
    if case.get("roi_motion"):
        ns = argparse.Namespace(roi_motion=True, roi=None, roi_min_area=None, roi_threshold=None)
        roi_cfg = build_roi_motion_config(cfg, ns)
        cfg["roi_motion"] = roi_cfg
        detector_name, object_detector = resolve_object_detector(cfg, roi_motion_force_enabled=True)
    elif case.get("object_detector"):
        backend_name = str(case["object_detector"]).strip().lower()
        backend_cfg_key = backend_name
        backend_cfg = dict(cfg.get(backend_cfg_key) or {})
        backend_cfg["enabled"] = True
        cfg[backend_cfg_key] = backend_cfg
        detector_name = backend_name
        object_detector = build_object_detector(backend_name, backend_cfg)
    else:
        detector_name, object_detector = resolve_object_detector(cfg)
    vision = VisionSensorFusion(detector, object_detector, ConstantPirReader())
    business = MockBusinessAdapter([BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True)] * 1000000)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"name": case["name"], "video": str(video), "pass": False, "errors": ["video_open_failed"]}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_dt = 1.0 / max(fps, 1e-6)
    base_ts = time.time()
    rows: list[dict[str, Any]] = []
    tick = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = base_ts + tick * frame_dt
        waiting_window_module.time.time = lambda ts=ts: ts
        packet = CameraPacket(rgb=frame, left_rgb=frame, timestamp=ts, meta={"video": str(video), "tick": tick})
        sensors = vision.read(packet)
        out = orchestrator.tick(RuntimeInput(business=business.read(), sensors=sensors, tick_id=tick))
        decision = out.actuator.raw_decision or out.fusion.decision.value
        rows.append(
            {
                "tick": tick,
                "time_sec": round(tick * frame_dt, 4),
                "marker_detected": sensors.marker is not None,
                "marker_id": sensors.marker.marker_id if sensors.marker else None,
                "decision": decision,
                "motor": out.actuator.motor.value,
                "objects": ";".join(obj.label for obj in sensors.objects),
                "roi_motion_triggered": getattr(object_detector, "last_triggered", None),
                "roi_motion_area": getattr(object_detector, "last_motion_area", None),
                "reason": out.actuator.reason,
            }
        )
        tick += 1
        if max_frames and tick >= max_frames:
            break
    cap.release()

    counts = _decision_counts(rows)
    detected = sum(1 for r in rows if r["marker_detected"])
    detection_rate = detected / len(rows) if rows else 0.0
    summary = {
        "name": case["name"],
        "video": str(video.relative_to(ROOT)),
        "environment": case.get("environment", "indoor"),
        "roi_motion": bool(case.get("roi_motion")),
        "frames": len(rows),
        "detected_frames": detected,
        "detection_rate": detection_rate,
        "decision_counts": counts,
        "errors": [],
        "pass": True,
    }

    exp = case.get("expect", {})
    def fail(msg: str) -> None:
        summary["pass"] = False
        summary["errors"].append(msg)

    if detection_rate < float(exp.get("detection_rate_min", 0.0)):
        fail(f"detection_rate {detection_rate:.3f} < {exp['detection_rate_min']}")
    if "detection_rate_max" in exp and detection_rate > float(exp["detection_rate_max"]):
        fail(f"detection_rate {detection_rate:.3f} > {exp['detection_rate_max']}")
    for decision, key in [("ALLOW_CART", "allow"), ("WARNING", "warning"), ("STOP", "stop"), ("IDLE", "idle")]:
        val = counts.get(decision, 0)
        if f"{key}_min" in exp and val < int(exp[f"{key}_min"]):
            fail(f"{decision} count {val} < {exp[f'{key}_min']}")
        if f"{key}_eq" in exp and val != int(exp[f"{key}_eq"]):
            fail(f"{decision} count {val} != {exp[f'{key}_eq']}")
    if exp.get("require_allow_stop_allow") and not _has_sequence(rows, ["ALLOW_CART", "STOP", "ALLOW_CART"]):
        fail("missing ALLOW_CART -> STOP -> ALLOW_CART sequence")
    if exp.get("require_missing_then_recover") and not _missing_then_recover(rows):
        fail("missing marker_missing -> recovered ALLOW_CART sequence")

    case_dir = output_dir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def run_acceptance(
    selected_names: list[str] | None = None,
    output_dir: Path | None = None,
    max_frames: int = 0,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Run the real-camera acceptance regression and return a summary dict.

    This is the importable entry point used both by the CLI and by the
    pytest slow-marker test, so day-to-day developers do not need to copy the
    case list anywhere else.
    """

    selected = CASES
    if selected_names:
        wanted = set(selected_names)
        selected = [c for c in CASES if c["name"] in wanted]
        missing = wanted - {c["name"] for c in selected}
        if missing:
            raise ValueError(f"unknown case(s): {sorted(missing)}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = ROOT / "captures" / "real_acceptance_runs" / run_id
    else:
        output_dir = Path(output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config.yaml")

    results = [replay_case(case, config, output_dir, max_frames) for case in selected]
    overall_pass = all(r.get("pass") for r in results)
    report = {"run_id": run_id, "pass": overall_pass, "cases": results, "output_dir": str(output_dir)}
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if print_summary:
        print("SafeZone real camera acceptance")
        print("=" * 60)
        for r in results:
            status = "PASS" if r.get("pass") else "FAIL"
            counts = r.get("decision_counts", {})
            print(
                f"[{status}] {r['name']} frames={r.get('frames')} det={r.get('detection_rate', 0):.3f} "
                f"IDLE={counts.get('IDLE',0)} WARNING={counts.get('WARNING',0)} "
                f"ALLOW={counts.get('ALLOW_CART',0)} STOP={counts.get('STOP',0)}"
            )
            for err in r.get("errors", []):
                print(f"       - {err}")
        print("-" * 60)
        print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
        print(f"summary: {output_dir / 'summary.json'}")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Run real recorded-camera SafeZone acceptance regression.")
    parser.add_argument("--output-dir", default=str(ROOT / "captures" / "real_acceptance_runs"))
    parser.add_argument("--case", action="append", help="run only named case; can repeat")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    report = run_acceptance(
        selected_names=args.case,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_frames=args.max_frames,
    )
    raise SystemExit(0 if report.get("pass") else 1)

if __name__ == "__main__":
    main()
