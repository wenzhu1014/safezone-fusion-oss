from __future__ import annotations

import argparse
from pathlib import Path

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--backend", default="MSMF", choices=["MSMF", "DSHOW", "ANY"])
    parser.add_argument("--output", default="projects/safezone_fusion/captures/camera_0_snapshot.jpg")
    args = parser.parse_args()

    backend_map = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW, "ANY": cv2.CAP_ANY}
    cap = cv2.VideoCapture(args.index, backend_map[args.backend])
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.index} {args.backend} not opened")
    # Warm up frames
    frame = None
    ok = False
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("failed to capture frame")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(out)
    print("shape", frame.shape)

if __name__ == "__main__":
    main()
