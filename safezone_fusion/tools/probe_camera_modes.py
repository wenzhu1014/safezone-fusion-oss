from __future__ import annotations

from pathlib import Path

import cv2

def fourcc_str(code: int) -> str:
    return "".join(chr((code >> 8 * i) & 255) for i in range(4))

def main():
    out_dir = Path("projects/safezone_fusion/captures")
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (1280, 480),
        (2560, 720),
    ]
    fourccs = ["MJPG", "YUY2", ""]
    backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
    for backend_name, backend in backends:
        print("\nBACKEND", backend_name)
        for width, height in requests:
            for fourcc in fourccs:
                cap = cv2.VideoCapture(0, backend)
                if not cap.isOpened():
                    continue
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                code = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_fourcc = fourcc_str(code)
                shape = getattr(frame, "shape", None)
                print(f"req={width}x{height} fourcc={fourcc or 'default'} ok={ok} actual={w}x{h} fps={fps} actual_fourcc={actual_fourcc!r} shape={shape}")
                if ok and frame is not None:
                    name = f"mode_{backend_name}_{width}x{height}_{fourcc or 'default'}_{int(w)}x{int(h)}.jpg"
                    cv2.imwrite(str(out_dir / name), frame)
                cap.release()

if __name__ == "__main__":
    main()
