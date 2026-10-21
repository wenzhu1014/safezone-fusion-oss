from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.opencv_capture import probe_camera

def main():
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes and save snapshots.")
    parser.add_argument("--max-index", type=int, default=5, help="probe indexes 0..max-index-1")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--snapshot-dir", default=str(ROOT / "captures" / "probe"))
    args = parser.parse_args()

    print("SafeZone Fusion camera probe")
    print("=" * 60)
    found = 0
    for index in range(args.max_index):
        result = probe_camera(index, width=args.width, height=args.height, snapshot_dir=args.snapshot_dir)
        if not result.opened:
            print(f"index {index}: not opened")
            continue
        found += 1
        print(
            f"index {index}: opened {result.width}x{result.height} "
            f"fps={result.fps:.1f} channels={result.channels} role={result.role_guess}"
        )
        if result.snapshot_path:
            print(f"  snapshot: {result.snapshot_path}")

    if found == 0:
        print("No camera opened. Check cable/permissions/other apps using the camera.")
    else:
        print(f"Found {found} camera(s). Current one-cable prototype can use --camera-index <index>.")

if __name__ == "__main__":
    main()
