from __future__ import annotations

from pathlib import Path

import cv2

def main():
    out = Path("projects/safezone_fusion/captures")
    out.mkdir(parents=True, exist_ok=True)
    for width, height in [(640, 480), (1280, 720), (1920, 1080), (1280, 480)]:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            print("not opened", flush=True)
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        shape = getattr(frame, "shape", None)
        print(f"req={width}x{height} ok={ok} actual={w}x{h} fps={fps} shape={shape}", flush=True)
        if ok and frame is not None:
            cv2.imwrite(str(out / f"hk1080_msmf_{width}x{height}_actual_{int(w)}x{int(h)}.jpg"), frame)
        cap.release()

if __name__ == "__main__":
    main()
