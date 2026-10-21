from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.adapters.mock_actuator import MockActuatorAdapter
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.interfaces import PirPacket, VisionSensorFusion
from safezone_fusion.src.vision.mock_pipeline import MockObjectDetector
from safezone_fusion.src.vision.object_detector_factory import resolve_object_detector, resolve_object_detectors
from safezone_fusion.src.safety_signal import SafetySignalEmitter, build_sinks_from_spec
from safezone_fusion.src.safety_watchdog import SafetyHeartbeat
from safezone_fusion.src.hud import HudState, render_hud, decision_to_status
from safezone_fusion.src.vision.opencv_capture import OpenCVCameraCapture, guess_frame_role
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector
from safezone_fusion.src.vision.lens_health import LensHealthMonitor, apply_lens_failsafe
from safezone_fusion.src.operating_state import OperatingStateMonitor, OperatingStateConfig

def apply_environment_profile(config, profile_name: str | None):
    if not profile_name:
        return
    profiles = config.get("environment_profiles", {})
    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(f"unknown environment profile: {profile_name}")
    waiting_window_patch = profile.get("waiting_window", {})
    config.setdefault("waiting_window", {}).update(waiting_window_patch)

class ConstantPirReader:
    def __init__(self, triggered: bool = False):
        self.triggered = triggered

    def read(self):
        return PirPacket(self.triggered, timestamp=time.time())

def _draw_text_panel(frame, lines, origin=(12, 24)):
    x, y = origin
    line_h = 22
    width = max(360, min(frame.shape[1] - 20, max(len(s) for s in lines) * 9 + 18)) if lines else 360
    height = line_h * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 18), (x - 8 + width, y - 18 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        color = (80, 255, 80) if i == 0 else (240, 240, 240)
        cv2.putText(frame, text, (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)

def draw_marker_debug(detector: OpenCVArucoMarkerDetector, frame, sensors=None, output=None, tick: int = 0, packet=None):
    if frame is None:
        return None
    out = frame.copy()
    # Draw ArUco boxes in the original camera orientation. The detector maps
    # MIRROR fallback corners back to original image coordinates, so the live
    # view does not suddenly flip when fallback mode toggles.
    if detector.last_corners is not None and detector.last_ids_array is not None:
        cv2.aruco.drawDetectedMarkers(out, detector.last_corners, detector.last_ids_array)

    marker_line = "marker: none"
    if sensors is not None and sensors.marker is not None:
        p = sensors.marker.pose
        marker_line = (
            f"marker: id={sensors.marker.marker_id} stable={sensors.marker.stable_frames} "
            f"speed={sensors.marker.speed_mps:.3f} smooth={sensors.marker.smoothed_speed_mps:.3f} "
            f"jitter={sensors.marker.center_jitter_px:.1f}px center_v={sensors.marker.center_speed_px_s:.1f}px/s "
            f"pose=({p.x:.2f},{p.y:.2f},{p.z:.2f})"
        )
    decision = "decision: n/a"
    reason = "reason: n/a"
    window = "window: n/a"
    motor = "motor: n/a"
    if output is not None:
        decision_value = output.actuator.raw_decision or output.fusion.decision.value
        decision = f"decision: {decision_value}"
        motor = f"motor: {output.actuator.motor.value}"
        reason = f"reason: {output.actuator.reason[:80]}"
        if output.fusion.waiting_window is not None:
            window = f"window: {output.fusion.waiting_window.reason[:80]}"
    lines = [
        f"SafeZone Live Overlay | tick={tick} | {'MIRROR' if detector.last_used_mirror else 'NORMAL'}",
    ]

    # Dual-camera health line. Only shown when a secondary camera is wired up.
    if packet is not None and packet.meta.get("secondary_index") is not None:
        skew = packet.meta.get("skew_ms")
        if skew is None:
            cam_line = (
                f"cams: primary={packet.meta.get('primary_index')} "
                f"secondary={packet.meta.get('secondary_index')} | skew=DROP"
            )
        else:
            cam_line = (
                f"cams: primary={packet.meta.get('primary_index')} "
                f"secondary={packet.meta.get('secondary_index')} | skew={skew:.1f}ms"
            )
        lines.append(cam_line)

    stream_name = getattr(detector, "last_stream_name", None) or "none"
    lines.extend([
        f"ids: {detector.last_detected_ids} | rejected: {detector.last_rejected_count} | stream={stream_name}",
        marker_line,
        decision,
        motor,
        window,
        reason,
    ])
    _draw_text_panel(out, lines)
    return out

def _parse_roi(value: str) -> list[float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 normalized floats")
    return parts

def build_roi_motion_config(config, args):
    roi_config = dict(config.get("roi_motion", {}))
    if args.roi_motion:
        roi_config["enabled"] = True
    if args.roi is not None:
        roi_config["roi"] = args.roi
    if args.roi_min_area is not None:
        roi_config["min_area_px"] = args.roi_min_area
    if args.roi_threshold is not None:
        roi_config["threshold"] = args.roi_threshold
    return roi_config

def main():
    parser = argparse.ArgumentParser(description="Live one-camera/dual-ready SafeZone Fusion demo.")
    parser.add_argument("--camera-index", type=int, default=0, help="primary camera index")
    parser.add_argument("--secondary-index", type=int, default=None, help="optional second camera index")
    parser.add_argument("--primary-role", default="auto", choices=["auto", "rgb", "nir", "mono", "color"])
    parser.add_argument("--secondary-role", default="auto", choices=["auto", "rgb", "nir", "mono", "color"])
    parser.add_argument("--auto", action="store_true", help="auto-detect cameras and assign RGB/NIR roles (overrides --camera-index etc); on-site foolproof startup")
    parser.add_argument("--auto-max-index", type=int, default=4, help="highest camera index to scan when --auto")
    parser.add_argument("--width", type=int, default=1280, help="requested capture width; default 1280 for single-camera 720p baseline")
    parser.add_argument("--height", type=int, default=720, help="requested capture height; default 720 for single-camera 720p baseline")
    parser.add_argument("--fps", type=float, default=30.0, help="requested capture FPS")
    parser.add_argument("--fourcc", default="MJPG", help="requested capture fourcc; use empty string to leave backend default")
    parser.add_argument("--marker-id", type=int, default=123)
    parser.add_argument("--any-marker", action="store_true", help="accept any ArUco id; useful for debugging")
    parser.add_argument("--environment", choices=["indoor", "outdoor"], default=None, help="apply environment profile before stability override")
    parser.add_argument("--relax-stability", action="store_true", help="prototype mode: stable_frames + ROI only; ignore speed/jitter stability")
    parser.add_argument(
        "--stability-mode",
        choices=["speed", "smoothed_speed", "jitter", "relaxed", "angled_marker", "anchored_position"],
        default=None,
        help="override waiting_window.stability_mode",
    )
    parser.add_argument("--marker-size-m", type=float, default=0.10)
    parser.add_argument("--no-window", action="store_true", help="print only; do not show OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Ctrl+C")
    parser.add_argument("--roi-motion", action="store_true", help="enable lightweight ROI motion detector as a provisional hand/obstacle detector")
    parser.add_argument("--roi", type=_parse_roi, default=None, help="override normalized ROI x0,y0,x1,y1 for --roi-motion")
    parser.add_argument("--roi-min-area", type=float, default=None, help="override minimum motion area in ROI to emit a hand object")
    parser.add_argument("--roi-threshold", type=int, default=None, help="override motion threshold for ROI detector")
    parser.add_argument("--roi-warmup", type=int, default=None, help="override warmup frames for ROI motion (NIR/AGC may need 30+)")
    parser.add_argument("--roi-min-consecutive", type=int, default=None, help="override min consecutive motion frames before STOP")
    parser.add_argument("--object-detector", type=str, default=None, help="override object detector backend name (mock|roi_motion|skin_motion|mediapipe_hand)")
    parser.add_argument("--object-detectors", type=str, default=None, help="comma-separated list of detectors to run together, e.g. 'mediapipe_hand,roi_motion' (semantic + motion safety-net). Overrides --object-detector when set.")
    parser.add_argument("--safety-signal", type=str, default=None, help="safety-signal output sink spec, e.g. 'stdout', 'latest:out/now.json', 'file:out/sig.jsonl', 'mqtt:127.0.0.1:1883:safezone/safety_signal'. Comma-separated for multiple. Sends STOP/ALLOW signals toward the Android/PLC side.")
    parser.add_argument("--safety-signal-audit", type=str, default=None, help="always-on JSONL audit log path for safety signals (durable record regardless of live transport)")
    parser.add_argument("--safety-heartbeat-sec", type=float, default=0.2, help="max seconds between safety signals even when decision is unchanged (keeps consumer watchdog alive)")
    parser.add_argument("--hud", action="store_true", help="overlay the product-grade HUD (big Chinese status banner + safety-zone highlight) for cabinet screen / demo")
    parser.add_argument("--hud-zone", type=_parse_roi, default=None, help="HUD safety-zone box as x0,y0,x1,y1 normalized (default: follow roi_motion roi)")
    parser.add_argument("--diag", type=str, default=None, help="per-frame diagnostic recorder output path (JSONL); records every layer's decision for offline 'why did it stop' analysis via tools/diag_summary.py")
    args = parser.parse_args()

    if args.auto:
        # On-site foolproof startup: scan cameras, assign RGB/NIR roles automatically.
        from safezone_fusion.src.vision.camera_autodetect import autodetect_plan, NoCameraError
        try:
            plan = autodetect_plan(max_index=args.auto_max_index, width=args.width, height=args.height)
        except NoCameraError as e:
            print(f"[--auto] ERROR: {e}")
            print("[--auto] 请检查摄像头是否插好 / 被其他程序占用。")
            return
        args.camera_index = plan.primary_index
        args.primary_role = plan.primary_role
        args.secondary_index = plan.secondary_index
        if plan.secondary_role:
            args.secondary_role = plan.secondary_role
        print(f"[--auto] {plan.note}")
        print(f"[--auto] 启用参数: {' '.join(plan.to_cli_args())}")

    config = load_config(ROOT / "config.yaml")
    apply_environment_profile(config, args.environment)
    if args.relax_stability:
        config.setdefault("waiting_window", {})["stability_mode"] = "relaxed"
    elif args.stability_mode:
        config.setdefault("waiting_window", {})["stability_mode"] = args.stability_mode
    orchestrator = SafeZoneOrchestrator(config)
    actuator = MockActuatorAdapter()
    detector = OpenCVArucoMarkerDetector(
        dictionary_name="DICT_6X6_250",
        marker_size_m=args.marker_size_m,
        target_marker_ids=None if args.any_marker else {args.marker_id},
    )
    roi_motion_config = build_roi_motion_config(config, args)
    if args.roi_motion or roi_motion_config.get("enabled"):
        config = dict(config)
        config["roi_motion"] = roi_motion_config
    if args.object_detectors:
        # Layered mode: run several detectors together (semantic + motion net).
        names = [n.strip() for n in args.object_detectors.split(",") if n.strip()]
        resolved_names, detectors = resolve_object_detectors(config, names)
        detector_name = "+".join(resolved_names)
        object_detector = detectors
        roi_motion_enabled = "roi_motion" in resolved_names
    else:
        if args.object_detector is not None:
            config = dict(config)
            config["object_detector"]["name"] = args.object_detector
        detector_name, object_detector = resolve_object_detector(config, roi_motion_force_enabled=args.roi_motion)
        roi_motion_enabled = detector_name == "roi_motion"
    vision = VisionSensorFusion(
        marker_detector=detector,
        object_detector=object_detector,
        pir_reader=ConstantPirReader(False),
    )
    business = MockBusinessAdapter([BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True)] * 1000000)

    # Pluggable safety-signal output toward the Android main board / PLC.
    safety_emitter = None
    safety_heartbeat = None
    if args.safety_signal or args.safety_signal_audit:
        safety_emitter = SafetySignalEmitter(
            build_sinks_from_spec(args.safety_signal, audit_path=args.safety_signal_audit),
            on_error=lambda sink, exc: print(f"[safety-signal] sink error: {exc}"),
        )
        # Emit at least every heartbeat interval even when the decision is
        # unchanged, so the consumer watchdog never mistakes calm for a dead
        # board. Changes always emit immediately.
        safety_heartbeat = SafetyHeartbeat(min_interval_sec=args.safety_heartbeat_sec)

    print("SafeZone Fusion v0.8 live camera demo")
    print("=" * 60)
    print(f"primary camera: {args.camera_index} role={args.primary_role}")
    print(f"capture request: {args.width}x{args.height}@{args.fps:g} fourcc={args.fourcc or 'default'}")
    if args.environment:
        print(f"environment: {args.environment}")
    print(f"stability_mode: {config.get('waiting_window', {}).get('stability_mode', 'speed')}")
    print(f"object_detector: {detector_name}")
    if roi_motion_enabled:
        print(
            f"roi_motion: roi={roi_motion_config.get('roi')} "
            f"min_area={float(roi_motion_config.get('min_area_px', 1200.0)):g} "
            f"threshold={roi_motion_config.get('threshold', 28)}"
        )
    if args.secondary_index is not None:
        print(f"secondary camera: {args.secondary_index} role={args.secondary_role}")
    print("Press q in the OpenCV window to exit." if not args.no_window else "Print-only mode.")

    tick = 0
    lens_monitor = LensHealthMonitor(config)
    op_state = OperatingStateMonitor(OperatingStateConfig(
        **(config.get("operating_state", {}) if isinstance(config.get("operating_state"), dict) else {})
    ))
    from safezone_fusion.src.diagnostics import DiagnosticRecorder, FrameDiag
    diag_recorder = DiagnosticRecorder(args.diag)
    if diag_recorder.enabled:
        print(f"[--diag] 逐帧诊断记录: {args.diag} (用 tools/diag_summary.py 分析)")
    with OpenCVCameraCapture(
        primary_index=args.camera_index,
        secondary_index=args.secondary_index,
        primary_role=args.primary_role,
        secondary_role=args.secondary_role,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc or None,
    ) as capture:
        while True:
            packet = capture.read()
            sensors = vision.read(packet)
            inp = RuntimeInput(business=business.read(), sensors=sensors, tick_id=tick)
            out = orchestrator.tick(inp)
            # Lens integrity self-check (IEC 61496 style): a covered/blinded/defocused
            # camera cannot be trusted, so force a fail-safe STOP independent of fusion.
            lens_frame = packet.rgb if packet.rgb is not None else packet.mono
            lens = lens_monitor.update(lens_frame)
            if lens.blinded:
                apply_lens_failsafe(out.actuator, lens)
            # Operating-state gate: STOP while warming up (background not learned)
            # or while FPS is degraded (safety distance was computed for a higher
            # FPS). Same fail-safe pattern as the lens self-check.
            op = op_state.tick()
            if op.stop_required:
                from safezone_fusion.src.interfaces import MotorAction
                out.actuator.motor = MotorAction.STOP
                out.actuator.reason = f"operating_state:{op.reason}"
                if isinstance(out.actuator.debug, dict):
                    out.actuator.debug["op_state"] = op.reason
            if safety_emitter is not None:
                key = (out.actuator.motor.value, out.actuator.reason)
                if safety_heartbeat is None or safety_heartbeat.should_emit(time.monotonic(), key):
                    safety_emitter.emit(out.actuator)

            marker_info = "none"
            if sensors.marker:
                p = sensors.marker.pose
                marker_info = (
                    f"id={sensors.marker.marker_id} stable={sensors.marker.stable_frames} "
                    f"pose=({p.x:.3f},{p.y:.3f},{p.z:.3f}) speed={sensors.marker.speed_mps:.3f} "
                    f"smooth={sensors.marker.smoothed_speed_mps:.3f} jitter={sensors.marker.center_jitter_px:.1f}px "
                    f"center_v={sensors.marker.center_speed_px_s:.1f}px/s"
                )
            role = guess_frame_role(packet.rgb if packet.rgb is not None else packet.mono)
            decision = out.actuator.raw_decision or out.fusion.decision.value
            debug_ids = detector.last_detected_ids
            mirror = " mirror" if detector.last_used_mirror else ""
            lens_info = "" if lens.reason == "lens_ok" else f" LENS={lens.reason}"
            if diag_recorder.enabled:
                d = FrameDiag(frame=tick, ts=time.time(), fps=op.fps)
                if sensors.marker:
                    d.marker_visible = bool(getattr(sensors.marker, "visible", True))
                    d.marker_coasting = bool(getattr(sensors.marker, "coasting", False))
                    d.marker_authorized = bool(getattr(sensors.marker, "authorized", False))
                    d.marker_stable_frames = getattr(sensors.marker, "stable_frames", None)
                    if getattr(sensors.marker, "center_px", None) is not None:
                        d.marker_center = tuple(sensors.marker.center_px)
                if sensors.objects:
                    for obj in sensors.objects:
                        op_pose = getattr(obj, "pose", None)
                        d.objects.append({
                            "label": obj.label,
                            "x": round(op_pose.x, 3) if op_pose else None,
                            "z": round(op_pose.z, 3) if op_pose else None,
                        })
                d.lens_blinded = bool(lens.blinded)
                d.lens_reason = lens.reason
                d.op_ready = op.ready
                d.op_degraded = op.degraded
                d.op_reason = op.reason
                d.decision = str(decision)
                d.reason = out.actuator.reason
                d.stop = (out.actuator.motor.value == "stop")
                diag_recorder.write(d)
            print(
                f"[{tick:04d}] role={role}{mirror} ids={debug_ids} rejected={detector.last_rejected_count} "
                f"marker={marker_info} decision={decision} motor={out.actuator.motor.value} reason={out.actuator.reason}{lens_info}"
            )

            if not args.no_window:
                frame = packet.rgb if packet.rgb is not None else packet.mono
                if frame is not None:
                    debug = draw_marker_debug(detector, frame, sensors=sensors, output=out, tick=tick, packet=packet)
                    if hasattr(vision, "draw_debug"):
                        debug = vision.draw_debug(debug)
                    # Draw detected object bounding boxes in green if available
                    if sensors and sensors.objects:
                        for obj in sensors.objects:
                            if obj.bbox_px is not None:
                                x0, y0, x1, y1 = obj.bbox_px
                                cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 0), 2)
                                cv2.putText(debug, obj.label, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if args.hud and debug is not None:
                        hh, ww = debug.shape[:2]
                        zr = args.hud_zone or roi_motion_config.get("roi") or [0.30, 0.62, 0.70, 0.95]
                        zone_px = (int(zr[0] * ww), int(zr[1] * hh), int(zr[2] * ww), int(zr[3] * hh))
                        status = decision_to_status(out.actuator.raw_decision, out.actuator.motor.value, out.actuator.reason)
                        debug = render_hud(debug, HudState(
                            status=status,
                            reason=out.actuator.reason,
                            safety_zone_px=zone_px,
                            blink_on=(tick // 6) % 2 == 0,  # ~blink on STOP
                            extra=f"stream={getattr(detector, 'last_stream_name', None) or 'none'}",
                        ))
                    if debug is not None:
                        cv2.imshow("safezone live camera", debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            tick += 1
            if args.max_frames and tick >= args.max_frames:
                break

    diag_recorder.close()
    if safety_emitter is not None:
        safety_emitter.close()
    if not args.no_window:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
