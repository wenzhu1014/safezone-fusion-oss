from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.opencv_capture import OpenCVCameraCapture, guess_frame_role

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def main() -> None:
    parser = argparse.ArgumentParser(description="Record short single-camera SafeZone acceptance clips.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--primary-role", default="auto", choices=["auto", "rgb", "nir", "mono", "color"])
    parser.add_argument("--width", type=int, default=1280, help="requested capture width; default 1280 for 720p baseline")
    parser.add_argument("--height", type=int, default=720, help="requested capture height; default 720 for 720p baseline")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="capture fourcc; use empty string for backend default")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--scenario",
        default="manual",
        help="scenario name, e.g. static_marker / slow_move_marker / sway_marker / marker_missing",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "captures" / "clips"))
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scenario}_{_timestamp()}_{args.width}x{args.height}.mp4"

    print("SafeZone camera clip recorder")
    print("=" * 60)
    print(f"camera={args.camera_index} request={args.width}x{args.height}@{args.fps:g} fourcc={args.fourcc or 'default'}")
    print(f"scenario={args.scenario} seconds={args.seconds:g}")
    print(f"output={out_path}")
    print("Press q in the preview window to stop early." if not args.no_window else "No-window mode.")

    writer = None
    frame_count = 0
    with OpenCVCameraCapture(
        primary_index=args.camera_index,
        primary_role=args.primary_role,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc or None,
    ) as capture:
        # Start timing after the camera has opened; UVC warm-up can take a few
        # seconds and should not consume the requested recording duration.
        start = time.time()
        while True:
            packet = capture.read()
            frame = packet.rgb if packet.rgb is not None else packet.mono
            if frame is None:
                raise RuntimeError("capture returned no frame")
            if writer is None:
                h, w = frame.shape[:2]
                # mp4v is widely writable on Windows/OpenCV. Capture fourcc can
                # still be MJPG; the saved clip is normalized to MP4 for replay.
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {out_path}")
                print(f"actual_frame={w}x{h} role={guess_frame_role(frame)}")
            writer.write(frame)
            frame_count += 1

            elapsed = time.time() - start
            if not args.no_window:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"REC {args.scenario} {elapsed:.1f}/{args.seconds:.1f}s frames={frame_count}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SafeZone clip recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if elapsed >= args.seconds:
                break

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    duration = time.time() - start if 'start' in locals() else 0.0
    print(f"saved={out_path}")
    print(f"frames={frame_count} duration_sec={duration:.2f}")

if __name__ == "__main__":
    main()
