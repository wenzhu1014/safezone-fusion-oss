"""Dual-camera probe: insert second cable, run this, see what shows up.

What this does (no algorithm, just plumbing):
  1. Probe a few candidate camera indices, report resolution / fps / channels / RGB-vs-mono guess.
  2. If at least two cameras are reachable, open them as a single OpenCVCameraCapture
     in dual-mode and read N frames.
  3. Report inter-camera skew (ms) per frame and a summary (min / mean / p95 / max).
  4. Save side-by-side snapshots so we can eyeball whether one is RGB and one is NIR.

Usage (after second cable is plugged in):

  python projects/safezone_fusion/tools/probe_dual_cameras.py
  python projects/safezone_fusion/tools/probe_dual_cameras.py --primary 0 --secondary 1 --frames 60
  python projects/safezone_fusion/tools/probe_dual_cameras.py --width 1280 --height 720 --no-snapshot

Exit status:
  0 = both cameras opened and produced frames
  1 = single camera or any failure
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.vision.opencv_capture import (  # noqa: E402
    OpenCVCameraCapture,
    probe_camera,
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-indices", default="0,1,2,3", help="comma-separated indices to probe")
    p.add_argument("--primary", type=int, default=None, help="explicit primary index (default: first opened)")
    p.add_argument("--secondary", type=int, default=None, help="explicit secondary index (default: second opened)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--frames", type=int, default=30, help="frames to read in dual mode")
    p.add_argument("--snapshot-dir", default="captures/dual_probe", help="where to write side-by-side preview")
    p.add_argument("--no-snapshot", action="store_true", help="skip saving snapshots")
    return p.parse_args()

def banner(s: str) -> None:
    print()
    print("=" * 64)
    print(s)
    print("=" * 64)

def probe_phase(indices, width, height, snapshot_dir):
    banner("PHASE 1 / probe candidate indices")
    results = []
    for idx in indices:
        r = probe_camera(idx, width=width, height=height, snapshot_dir=snapshot_dir)
        results.append(r)
        if r.opened:
            print(
                f"  index={r.index:<2} OPEN  {r.width}x{r.height} @ {r.fps:.1f}fps  "
                f"channels={r.channels}  role_guess={r.role_guess}  "
                f"snapshot={r.snapshot_path}"
            )
        else:
            print(f"  index={r.index:<2} (none)")
    opened = [r for r in results if r.opened]
    return opened

def dual_phase(primary_idx, secondary_idx, args):
    banner(f"PHASE 2 / dual-capture primary={primary_idx} secondary={secondary_idx}")
    cam = OpenCVCameraCapture(
        primary_index=primary_idx,
        secondary_index=secondary_idx,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    skews = []
    primary_ok = secondary_ok = 0
    last_rgb = last_nir = None
    try:
        for i in range(args.frames):
            packet = cam.read()
            if packet.rgb is not None:
                primary_ok += 1
                last_rgb = packet.rgb
            if packet.nir is not None:
                secondary_ok += 1
                last_nir = packet.nir
            skew = packet.meta.get("skew_ms")
            if skew is not None:
                skews.append(skew)
            if i < 5 or i == args.frames - 1:
                print(
                    f"  frame {i:>3}  ts_p={packet.meta['ts_primary']:.4f}  "
                    f"ts_s={packet.meta['ts_secondary']}  skew_ms={skew}  "
                    f"rgb={'Y' if packet.rgb is not None else '-'}  "
                    f"nir={'Y' if packet.nir is not None else '-'}"
                )
            time.sleep(1.0 / max(args.fps, 1.0))
    finally:
        cam.release()

    banner("PHASE 3 / summary")
    print(f"  frames requested : {args.frames}")
    print(f"  primary frames   : {primary_ok}")
    print(f"  secondary frames : {secondary_ok}")
    if skews:
        print(
            f"  skew_ms (n={len(skews)}): "
            f"min={min(skews):.2f}  mean={statistics.fmean(skews):.2f}  "
            f"p95={sorted(skews)[max(0, int(len(skews)*0.95)-1)]:.2f}  "
            f"max={max(skews):.2f}"
        )
    else:
        print("  skew_ms: (no dual-frame samples — secondary may have dropped)")

    # Side-by-side snapshot
    if not args.no_snapshot and last_rgb is not None and last_nir is not None:
        out_dir = Path(args.snapshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # resize secondary to match primary if needed
            h, w = last_rgb.shape[:2]
            sec = last_nir
            if sec.shape[:2] != (h, w):
                sec = cv2.resize(sec, (w, h))
            if len(sec.shape) == 2:
                sec = cv2.cvtColor(sec, cv2.COLOR_GRAY2BGR)
            side = cv2.hconcat([last_rgb, sec])
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = out_dir / f"dual_probe_{primary_idx}_{secondary_idx}_{stamp}.jpg"
            cv2.imwrite(str(out), side)
            print(f"  side-by-side snapshot: {out}")
        except Exception as e:
            print(f"  [warn] side-by-side snapshot failed: {e}")

    return primary_ok > 0 and secondary_ok > 0

def main():
    args = parse_args()
    indices = [int(x) for x in args.candidate_indices.split(",") if x.strip()]
    snapshot_dir = None if args.no_snapshot else args.snapshot_dir

    opened = probe_phase(indices, args.width, args.height, snapshot_dir)
    if len(opened) == 0:
        print("\n[FAIL] no cameras opened. Plug in cable or check indices.")
        return 1

    primary_idx = args.primary if args.primary is not None else opened[0].index
    if len(opened) < 2:
        print(f"\n[INFO] only one camera reachable (index={primary_idx}). Skipping dual phase.")
        print("       Plug in the second cable and re-run to see dual-capture skew.")
        return 1

    secondary_idx = args.secondary
    if secondary_idx is None:
        for r in opened:
            if r.index != primary_idx:
                secondary_idx = r.index
                break

    ok = dual_phase(primary_idx, secondary_idx, args)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
