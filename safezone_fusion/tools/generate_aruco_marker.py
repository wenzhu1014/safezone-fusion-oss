from __future__ import annotations

import argparse
from pathlib import Path

import cv2

def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is unavailable. Install opencv-contrib-python.")
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id)

def generate(marker_id: int, output: Path, dictionary_name: str = "DICT_6X6_250", size_px: int = 800, border_bits: int = 1) -> Path:
    dictionary = get_dictionary(dictionary_name)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px, borderBits=border_bits)
    # Real printed markers need a quiet white zone around the black marker.
    margin = max(size_px // 5, 40)
    image = 255 * __import__('numpy').ones((size_px + margin * 2, size_px + margin * 2), dtype='uint8')
    image[margin:margin + size_px, margin:margin + size_px] = marker
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"Failed to write marker image: {output}")
    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image.")
    parser.add_argument("--id", type=int, default=1023, help="Marker id")
    parser.add_argument("--dict", default="DICT_6X6_250", help="OpenCV ArUco dictionary name")
    parser.add_argument("--size", type=int, default=800, help="Image size in pixels")
    parser.add_argument("--border", type=int, default=1, help="Border bits")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path("projects/safezone_fusion/generated_markers") / f"aruco_{args.dict}_{args.id}.png"
    path = generate(args.id, output, args.dict, args.size, args.border)
    print(path)

if __name__ == "__main__":
    main()
