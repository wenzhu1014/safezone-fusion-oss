from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.stereo_splitter import StereoFrameSplitter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout", choices=["single", "side_by_side", "top_bottom"], default="top_bottom")
    parser.add_argument("--out-dir", default="projects/safezone_fusion/captures/splits")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"failed to read image: {args.image}")
    splitter = StereoFrameSplitter(args.layout)
    result = splitter.split(image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_path = out_dir / f"{Path(args.image).stem}_{args.layout}_left.jpg"
    cv2.imwrite(str(left_path), result.left_rgb)
    print("layout", result.layout, result.reason)
    print("left", left_path)
    if result.right_rgb is not None:
        right_path = out_dir / f"{Path(args.image).stem}_{args.layout}_right.jpg"
        cv2.imwrite(str(right_path), result.right_rgb)
        print("right", right_path)

if __name__ == "__main__":
    main()
