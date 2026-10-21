from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.opencv_capture import OpenCVCameraCapture, guess_frame_role

@dataclass
class PhaseStats:
    name: str
    samples: int
    detected: int
    detection_rate: float
    center_speed_px_s_p50: float | None
    center_speed_px_s_p90: float | None
    center_speed_px_s_p95: float | None
    center_speed_px_s_max: float | None
    jitter_px_p50: float | None
    jitter_px_p90: float | None
    jitter_px_p95: float | None
    jitter_px_max: float | None

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)

def phase_stats(name: str, rows: list[dict]) -> PhaseStats:
    detected_rows = [r for r in rows if r.get("detected")]
    speeds = [float(r["center_speed_px_s"]) for r in detected_rows]
    jitters = [float(r["center_jitter_px"]) for r in detected_rows]
    return PhaseStats(
        name=name,
        samples=len(rows),
        detected=len(detected_rows),
        detection_rate=(len(detected_rows) / len(rows)) if rows else 0.0,
        center_speed_px_s_p50=percentile(speeds, 0.50),
        center_speed_px_s_p90=percentile(speeds, 0.90),
        center_speed_px_s_p95=percentile(speeds, 0.95),
        center_speed_px_s_max=max(speeds) if speeds else None,
        jitter_px_p50=percentile(jitters, 0.50),
        jitter_px_p90=percentile(jitters, 0.90),
        jitter_px_p95=percentile(jitters, 0.95),
        jitter_px_max=max(jitters) if jitters else None,
    )

def recommendation(still: PhaseStats, slow_move: PhaseStats | None, safety_margin: float) -> dict:
    if still.detected == 0:
        return {
            "angled_still_center_speed_max_px_s": None,
            "angled_still_time_sec": None,
            "stable_center_speed_max_px_s": None,
            "notes": [
                "WARNING: static phase detected 0 markers; put the ArUco marker in view and rerun calibration before changing config"
            ],
        }
    still_p95 = still.center_speed_px_s_p95 or 0.0
    still_max = still.center_speed_px_s_max or still_p95
    # Keep enough room above real static noise, but do not exceed the slow-move
    # distribution if the user captured a movement phase.
    noise_floor = max(still_p95 * safety_margin, still_max * 1.05, 1.0)
    rec_speed = noise_floor
    notes = [f"static noise p95={still_p95:.2f}px/s max={still_max:.2f}px/s"]
    if slow_move and slow_move.center_speed_px_s_p50 is not None:
        slow_p50 = slow_move.center_speed_px_s_p50
        slow_p10 = None
        # Approximate p10 from the raw p50/p90 relation is not reliable, so use a
        # conservative cap below slow p50. Real review should inspect CSV/JSON.
        cap = max(1.0, slow_p50 * 0.75)
        rec_speed = min(rec_speed, cap) if rec_speed < cap else cap
        notes.append(f"slow-move p50={slow_p50:.2f}px/s; cap threshold around 75% of p50")
        if rec_speed <= still_p95:
            notes.append("WARNING: static noise overlaps slow movement; improve lighting/marker size/camera angle before tightening threshold")
    return {
        "angled_still_center_speed_max_px_s": round(float(rec_speed), 2),
        "angled_still_time_sec": 0.8,
        "stable_center_speed_max_px_s": round(float(max(rec_speed * 2.5, 10.0)), 2),
        "notes": notes,
    }

def sway_recommendation(sway: PhaseStats | None, safety_margin: float) -> dict:
    if sway is None or sway.detected == 0:
        return {
            "wind_sway_anchor_drift_max_px": None,
            "wind_sway_jitter_max_px": None,
            "wind_sway_min_anchor_age_sec": None,
            "notes": ["sway phase not captured; keep current wind_sway defaults"],
        }
    jitter_p95 = sway.jitter_px_p95 or 0.0
    jitter_max = sway.jitter_px_max or jitter_p95
    # center_jitter_px is the max radius around the recent center window. Use it
    # as a practical proxy for local wind/structure sway. Anchor drift should be
    # looser than local jitter, but still small enough to reject real translation.
    rec_jitter = max(1.0, jitter_p95 * safety_margin)
    rec_anchor = max(3.0, min(18.0, max(jitter_max * safety_margin, rec_jitter * 2.5)))
    return {
        "wind_sway_anchor_drift_max_px": round(float(rec_anchor), 2),
        "wind_sway_jitter_max_px": round(float(rec_jitter), 2),
        "wind_sway_min_anchor_age_sec": 0.3,
        "notes": [
            f"sway jitter p95={jitter_p95:.2f}px max={jitter_max:.2f}px",
            "anchor drift is capped at 18px; if real wind exceeds this, prefer longer stable_time or pause docking",
        ],
    }

def countdown(label: str, seconds: int) -> None:
    print(f"\n{label}")
    for i in range(seconds, 0, -1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  recording now.          ")

def capture_phase(capture: OpenCVCameraCapture, detector: OpenCVArucoMarkerDetector, name: str, duration_sec: float, show_window: bool) -> list[dict]:
    import cv2

    rows: list[dict] = []
    start = time.time()
    tick = 0
    while True:
        now = time.time()
        if now - start >= duration_sec:
            break
        packet = capture.read()
        marker = detector.detect(packet)
        frame = packet.rgb if packet.rgb is not None else packet.mono
        role = guess_frame_role(frame)
        row = {
            "phase": name,
            "tick": tick,
            "timestamp": packet.timestamp,
            "role": role,
            "detected": marker is not None,
            "used_mirror": detector.last_used_mirror,
            "ids": json.dumps(detector.last_detected_ids, ensure_ascii=False),
            "rejected": detector.last_rejected_count,
        }
        if marker is not None:
            row.update(
                {
                    "marker_id": marker.marker_id,
                    "stable_frames": marker.stable_frames,
                    "center_x": marker.center_px[0] if marker.center_px else None,
                    "center_y": marker.center_px[1] if marker.center_px else None,
                    "center_speed_px_s": marker.center_speed_px_s,
                    "center_jitter_px": marker.center_jitter_px,
                    "speed_mps": marker.speed_mps,
                    "smoothed_speed_mps": marker.smoothed_speed_mps,
                }
            )
        else:
            row.update(
                {
                    "marker_id": None,
                    "stable_frames": 0,
                    "center_x": None,
                    "center_y": None,
                    "center_speed_px_s": None,
                    "center_jitter_px": None,
                    "speed_mps": None,
                    "smoothed_speed_mps": None,
                }
            )
        rows.append(row)

        if show_window and frame is not None:
            out = frame.copy()
            if detector.last_corners is not None and detector.last_ids_array is not None:
                cv2.aruco.drawDetectedMarkers(out, detector.last_corners, detector.last_ids_array)
            label = f"{name} {now - start:.1f}/{duration_sec:.1f}s ids={detector.last_detected_ids}"
            if marker is not None:
                label += f" center_v={marker.center_speed_px_s:.1f}px/s jitter={marker.center_jitter_px:.1f}px"
            cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 2, cv2.LINE_AA)
            cv2.imshow("SafeZone marker stability calibration", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        tick += 1
    return rows

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SafeZone ArUco marker stability thresholds from live camera data.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--primary-role", default="auto", choices=["auto", "rgb", "nir", "mono", "color"])
    parser.add_argument("--width", type=int, default=1280, help="requested capture width; default 1280 for 720p baseline")
    parser.add_argument("--height", type=int, default=720, help="requested capture height; default 720 for 720p baseline")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="requested capture fourcc; use empty string to leave backend default")
    parser.add_argument("--marker-id", type=int, default=123)
    parser.add_argument("--any-marker", action="store_true")
    parser.add_argument("--static-sec", type=float, default=5.0, help="seconds to hold marker completely still")
    parser.add_argument("--slow-sec", type=float, default=5.0, help="seconds to move marker slowly; set 0 to skip")
    parser.add_argument("--sway-sec", type=float, default=0.0, help="seconds to gently sway marker/cabinet in place; set 0 to skip")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "calibration_runs"))
    parser.add_argument("--safety-margin", type=float, default=1.5)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"marker_stability_{run_id}"
    detector = OpenCVArucoMarkerDetector(target_marker_ids=None if args.any_marker else {args.marker_id})

    print("SafeZone marker stability calibration")
    print("=" * 60)
    print(f"camera={args.camera_index} marker_id={'any' if args.any_marker else args.marker_id}")
    print(f"output={out_dir}")
    print("Phase 1: keep the marker completely still.")
    if args.sway_sec > 0:
        print("Phase 2: gently sway the marker/cabinet in place, like outdoor wind.")
    if args.slow_sec > 0:
        print("Phase 3: move the marker very slowly, like the worst-case creeping cart.")
    print("Press q in the preview window to abort a phase." if not args.no_window else "No-window mode.")

    all_rows: list[dict] = []
    with OpenCVCameraCapture(
        primary_index=args.camera_index,
        primary_role=args.primary_role,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc or None,
    ) as capture:
        countdown("STATIC phase: keep marker still", args.countdown)
        static_rows = capture_phase(capture, detector, "static", args.static_sec, show_window=not args.no_window)
        all_rows.extend(static_rows)

        sway_rows: list[dict] = []
        if args.sway_sec > 0:
            countdown("SWAY phase: gently sway marker/cabinet in place", args.countdown)
            sway_rows = capture_phase(capture, detector, "sway", args.sway_sec, show_window=not args.no_window)
            all_rows.extend(sway_rows)

        slow_rows: list[dict] = []
        if args.slow_sec > 0:
            countdown("SLOW_MOVE phase: move marker slowly", args.countdown)
            slow_rows = capture_phase(capture, detector, "slow_move", args.slow_sec, show_window=not args.no_window)
            all_rows.extend(slow_rows)

    if not args.no_window:
        import cv2

        cv2.destroyAllWindows()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "samples.csv"
    write_csv(csv_path, all_rows)

    static_stats = phase_stats("static", static_rows)
    sway_stats = phase_stats("sway", sway_rows) if sway_rows else None
    slow_stats = phase_stats("slow_move", slow_rows) if slow_rows else None
    rec = recommendation(static_stats, slow_stats, args.safety_margin)
    sway_rec = sway_recommendation(sway_stats, args.safety_margin)
    summary = {
        "run_id": run_id,
        "camera_index": args.camera_index,
        "marker_id": None if args.any_marker else args.marker_id,
        "static": asdict(static_stats),
        "sway": asdict(sway_stats) if sway_stats else None,
        "slow_move": asdict(slow_stats) if slow_stats else None,
        "recommendation": rec,
        "sway_recommendation": sway_rec,
        "csv": str(csv_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSuggested config snippet:")
    if rec["angled_still_center_speed_max_px_s"] is None:
        print("  calibration incomplete: no marker detected in static phase")
    else:
        print("waiting_window:")
        print(f"  angled_still_center_speed_max_px_s: {rec['angled_still_center_speed_max_px_s']}")
        print(f"  angled_still_time_sec: {rec['angled_still_time_sec']}")
        if sway_rec["wind_sway_anchor_drift_max_px"] is not None:
            print(f"  wind_sway_anchor_drift_max_px: {sway_rec['wind_sway_anchor_drift_max_px']}")
            print(f"  wind_sway_jitter_max_px: {sway_rec['wind_sway_jitter_max_px']}")
            print(f"  wind_sway_min_anchor_age_sec: {sway_rec['wind_sway_min_anchor_age_sec']}")
    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()
