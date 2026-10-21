from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import cv2

from .interfaces import CameraPacket

@dataclass
class CameraProbeResult:
    index: int
    opened: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    channels: int = 0
    role_guess: str = "unknown"
    snapshot_path: str | None = None

def _frame_channels(frame) -> int:
    if frame is None:
        return 0
    if len(frame.shape) == 2:
        return 1
    return int(frame.shape[2])

def guess_frame_role(frame) -> str:
    """Best-effort role guess: rgb/color vs mono/nir.

    OpenCV often returns 3-channel BGR even for mono UVC devices, so do not rely
    only on channel count. Compare B/G/R channel differences to detect grayscale-like frames.
    """

    if frame is None:
        return "unknown"
    if len(frame.shape) == 2:
        return "mono_or_nir"
    if len(frame.shape) == 3 and frame.shape[2] >= 3:
        b, g, r = cv2.split(frame[:, :, :3])
        diff = float(cv2.absdiff(b, g).mean() + cv2.absdiff(g, r).mean() + cv2.absdiff(b, r).mean()) / 3.0
        if diff < 2.0:
            return "mono_or_nir"
        return "rgb"
    return "unknown"

def probe_camera(index: int, width: int | None = None, height: int | None = None, snapshot_dir: str | Path | None = None) -> CameraProbeResult:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        cap.release()
        return CameraProbeResult(index=index, opened=False)

    # Warm up a little; some UVC cameras return black/old frames first.
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    channels = _frame_channels(frame)
    role = guess_frame_role(frame)
    snapshot_path = None

    if frame is not None and snapshot_dir is not None:
        d = Path(snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        snapshot_path = str(d / f"camera_{index}_{role}_{actual_w}x{actual_h}.jpg")
        cv2.imwrite(snapshot_path, frame)

    cap.release()
    return CameraProbeResult(index=index, opened=True, width=actual_w, height=actual_h, fps=fps, channels=channels, role_guess=role, snapshot_path=snapshot_path)

class OpenCVCameraCapture:
    """Single-first, dual-ready OpenCV capture.

    Current prototype can run with only one cable/camera. If secondary_index is
    provided later, the same reader fills both rgb/nir/mono slots automatically.
    """

    def __init__(
        self,
        primary_index: int = 0,
        secondary_index: int | None = None,
        primary_role: str = "auto",
        secondary_role: str = "auto",
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = "MJPG",
        backend: int = cv2.CAP_DSHOW,
    ):
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.primary_role = primary_role
        self.secondary_role = secondary_role
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.backend = backend
        self.primary = self._open(primary_index)
        self.secondary = self._open(secondary_index) if secondary_index is not None else None

    def _open(self, index: int | None):
        if index is None:
            return None
        cap = cv2.VideoCapture(index, self.backend)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"failed to open camera index {index}")
        return cap

    def read(self) -> CameraPacket:
        # Read both cameras as close together as possible. We sample the wall clock
        # before each grab() so we can later monitor inter-camera skew. We also
        # use grab()->retrieve() rather than back-to-back read() so the second
        # camera's exposure starts only ~ms after the first, not a full decode
        # later. Falls back to read() if the backend does not support grab/retrieve.
        frames: list[tuple[str, object]] = []

        ts_primary = time.time()
        if not self.primary.grab():
            raise RuntimeError(f"failed to grab primary camera index {self.primary_index}")
        ts_secondary: float | None = None
        if self.secondary is not None:
            ts_secondary = time.time()
            self.secondary.grab()  # tolerate transient grab failure; retrieve below will skip

        ok, frame = self.primary.retrieve()
        if not ok or frame is None:
            raise RuntimeError(f"failed to retrieve primary camera index {self.primary_index}")
        frames.append((self.primary_role, frame))

        if self.secondary is not None:
            ok2, frame2 = self.secondary.retrieve()
            if ok2 and frame2 is not None:
                frames.append((self.secondary_role, frame2))
            else:
                # Lost a single secondary frame is not fatal; downstream sees only primary.
                ts_secondary = None

        ts = ts_primary
        skew_ms = None
        if ts_secondary is not None:
            skew_ms = round((ts_secondary - ts_primary) * 1000.0, 2)
        packet = CameraPacket(
            timestamp=ts,
            meta={
                "primary_index": self.primary_index,
                "secondary_index": self.secondary_index,
                "ts_primary": ts_primary,
                "ts_secondary": ts_secondary,
                "skew_ms": skew_ms,
            },
        )
        for role, frame in frames:
            r = guess_frame_role(frame) if role == "auto" else role
            if r in ("rgb", "color"):
                if packet.rgb is None:
                    packet.rgb = frame
                else:
                    packet.left_rgb = packet.left_rgb or frame
            elif r in ("nir", "mono", "mono_or_nir"):
                if packet.nir is None:
                    packet.nir = frame
                if packet.mono is None:
                    packet.mono = frame
            else:
                # Unknown single stream: expose it as rgb for algorithms that expect color,
                # and as mono too if it is grayscale-like.
                if packet.rgb is None:
                    packet.rgb = frame
                if packet.mono is None and guess_frame_role(frame) == "mono_or_nir":
                    packet.mono = frame

        # Compatibility with old left/right APIs.
        if packet.left_rgb is None and packet.rgb is not None:
            packet.left_rgb = packet.rgb
        # NOTE: do NOT mirror packet.nir into right_rgb. right_rgb is meant for a
        # second *color* camera (stereo RGB). Copying NIR here gave it a 'right_rgb'
        # alias that the ArUco candidate order (rgb->left_rgb->right_rgb->mono->nir)
        # picked up BEFORE the real nir slot, so NIR was used under a wrong name and
        # looked like "two RGB cameras fighting over the marker" (observed 2026-06-22).
        # Only alias right_rgb from an actual second RGB stream.
        return packet

    def release(self) -> None:
        if self.primary is not None:
            self.primary.release()
        if self.secondary is not None:
            self.secondary.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
