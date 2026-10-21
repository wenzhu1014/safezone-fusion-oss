from __future__ import annotations

"""Replay a recorded camera clip through the SafeZone pipeline.

⚠️ NOTE on NIR / dual-camera testing:
    Recorded clips reflect ONLY the RGB (or whatever was the primary)
    capture path at recording time. They cannot be used to validate the
    NIR fallback / mono path of the marker detector, because:
      - NIR cameras see infrared, not visible light;
      - LCD/OLED screens look essentially blank or smeared in NIR;
      - so replaying a video on screen does not exercise NIR optics at all.
    NIR / dual-camera regression must be done with real cameras + physical
    printed markers. See NOTES.md sections 2026-06-18 10:25 and 11:10.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.demo_live_camera import apply_environment_profile, build_roi_motion_config, draw_marker_debug
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.interfaces import CameraPacket, PirPacket, VisionSensorFusion
from safezone_fusion.src.vision.mock_pipeline import MockObjectDetector
from safezone_fusion.src.vision.object_detector_factory import resolve_object_detector
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector
import safezone_fusion.src.waiting_window as waiting_window_module

class ConstantPirReader:
    def __init__(self, triggered: bool = False):
        self.triggered = triggered

    def read(self):
        return PirPacket(self.triggered, timestamp=time.time())

def _safe_text(value) -> str:
    return "" if value is None else str(value)

def _parse_roi(value: str) -> list[float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 normalized floats")
    return parts

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded SafeZone camera clip through marker/fusion logic.")
    parser.add_argument("video", help="recorded clip path")
    parser.add_argument("--marker-id", type=int, default=123)
    parser.add_argument("--any-marker", action="store_true")
    parser.add_argument("--environment", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument(
        "--stability-mode",
        choices=["speed", "smoothed_speed", "jitter", "relaxed", "angled_marker", "anchored_position"],
        default=None,
        help="override waiting_window.stability_mode after environment profile",
    )
    parser.add_argument("--marker-size-m", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(ROOT / "captures" / "replay_runs"))
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--roi-motion", action="store_true", help="enable lightweight ROI motion detector as a provisional hand/obstacle detector")
    parser.add_argument("--roi", type=_parse_roi, default=None, help="override normalized ROI x0,y0,x1,y1 for --roi-motion")
    parser.add_argument("--roi-min-area", type=float, default=None)
    parser.add_argument("--roi-threshold", type=int, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"{video_path.stem}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frames.csv"
    summary_path = out_dir / "summary.json"

    config = load_config(ROOT / "config.yaml")
    apply_environment_profile(config, args.environment)
    if args.stability_mode:
        config.setdefault("waiting_window", {})["stability_mode"] = args.stability_mode

    orchestrator = SafeZoneOrchestrator(config)
    detector = OpenCVArucoMarkerDetector(
        dictionary_name="DICT_6X6_250",
        marker_size_m=args.marker_size_m,
        target_marker_ids=None if args.any_marker else {args.marker_id},
    )
    roi_motion_config = build_roi_motion_config(config, args)
    if args.roi_motion or roi_motion_config.get("enabled"):
        config["roi_motion"] = roi_motion_config
    detector_name, object_detector = resolve_object_detector(config, roi_motion_force_enabled=args.roi_motion)
    roi_motion_enabled = detector_name == "roi_motion"
    vision = VisionSensorFusion(
        marker_detector=detector,
        object_detector=object_detector,
        pir_reader=ConstantPirReader(False),
    )
    business = MockBusinessAdapter([BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True)] * 1000000)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_dt = 1.0 / max(fps, 1e-6)

    print("SafeZone clip replay")
    print("=" * 60)
    print(f"video={video_path}")
    print(f"environment={args.environment} stability_mode={config.get('waiting_window', {}).get('stability_mode')}")
    print(f"object_detector={detector_name}")
    if roi_motion_enabled:
        print(
            f"roi_motion=on roi={roi_motion_config.get('roi')} "
            f"min_area={float(roi_motion_config.get('min_area_px', 1200.0)):g} "
            f"threshold={roi_motion_config.get('threshold', 28)}"
        )
    print(f"output={out_dir}")

    rows: list[dict] = []
    decision_counts: dict[str, int] = {}
    motor_counts: dict[str, int] = {}
    detected_frames = 0
    allowed_frames = 0
    stopped_frames = 0
    warned_frames = 0

    tick = 0
    base_ts = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = base_ts + tick * frame_dt
        packet = CameraPacket(rgb=frame, left_rgb=frame, timestamp=ts, meta={"video": str(video_path), "tick": tick})
        # Replay should use video time, not wall-clock processing speed. The
        # waiting-window checker has timers based on time.time(), so patch its
        # module-local time source for this frame before running the decision.
        waiting_window_module.time.time = lambda ts=ts: ts
        sensors = vision.read(packet)
        out = orchestrator.tick(RuntimeInput(business=business.read(), sensors=sensors, tick_id=tick))
        decision = out.actuator.raw_decision or out.fusion.decision.value
        motor = out.actuator.motor.value
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        motor_counts[motor] = motor_counts.get(motor, 0) + 1
        if sensors.marker is not None:
            detected_frames += 1
        if decision == "ALLOW_CART":
            allowed_frames += 1
        elif decision == "STOP":
            stopped_frames += 1
        elif decision == "WARNING":
            warned_frames += 1

        marker = sensors.marker
        row = {
            "tick": tick,
            "time_sec": round(tick * frame_dt, 4),
            "ids": json.dumps(detector.last_detected_ids, ensure_ascii=False),
            "used_mirror": detector.last_used_mirror,
            "rejected": detector.last_rejected_count,
            "marker_detected": marker is not None,
            "marker_id": marker.marker_id if marker else None,
            "stable_frames": marker.stable_frames if marker else 0,
            "center_x": marker.center_px[0] if marker and marker.center_px else None,
            "center_y": marker.center_px[1] if marker and marker.center_px else None,
            "center_speed_px_s": marker.center_speed_px_s if marker else None,
            "center_jitter_px": marker.center_jitter_px if marker else None,
            "pose_x": marker.pose.x if marker else None,
            "pose_y": marker.pose.y if marker else None,
            "pose_z": marker.pose.z if marker else None,
            "decision": decision,
            "motor": motor,
            "objects": ";".join(obj.label for obj in sensors.objects),
            "roi_motion_area": getattr(object_detector, "last_motion_area", None),
            "roi_motion_triggered": getattr(object_detector, "last_triggered", None),
            "waiting_window": out.fusion.waiting_window.reason if out.fusion.waiting_window else None,
            "reason": out.actuator.reason,
        }
        rows.append(row)

        print(
            f"[{tick:04d}] ids={detector.last_detected_ids} marker={row['marker_id'] or 'none'} "
            f"decision={decision} motor={motor} reason={_safe_text(out.actuator.reason)[:100]}"
        )

        if not args.no_window:
            debug = draw_marker_debug(detector, frame, sensors=sensors, output=out, tick=tick)
            if roi_motion_enabled and hasattr(object_detector, "draw_debug"):
                debug = object_detector.draw_debug(debug)
            if debug is not None:
                cv2.imshow("SafeZone clip replay", debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        tick += 1
        if args.max_frames and tick >= args.max_frames:
            break

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "video": str(video_path),
        "run_id": run_id,
        "frames": len(rows),
        "fps": fps,
        "environment": args.environment,
        "stability_mode": config.get("waiting_window", {}).get("stability_mode"),
        "marker_id": None if args.any_marker else args.marker_id,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / len(rows) if rows else 0.0,
        "decision_counts": decision_counts,
        "motor_counts": motor_counts,
        "allowed_frames": allowed_frames,
        "warned_frames": warned_frames,
        "stopped_frames": stopped_frames,
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    main()
