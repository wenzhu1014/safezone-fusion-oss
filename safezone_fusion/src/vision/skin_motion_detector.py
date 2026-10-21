"""Skin + motion fused hand detector.

This is a lightweight upgrade over :class:`RoiMotionObjectDetector` that adds a
skin-tone gating step in HSV space, so things like display flicker, wind sway
of paper, or general background motion are less likely to trigger ``STOP``.
It still runs entirely with OpenCV/Numpy and does not need MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..models import ObjectObservation, Pose
from .interfaces import CameraPacket

@dataclass
class SkinMotionConfig:
    enabled: bool = False
    roi: tuple[float, float, float, float] = (0.30, 0.62, 0.70, 0.95)
    min_area_px: float = 1500.0
    motion_threshold: int = 28
    blur_kernel: int = 5
    warmup_frames: int = 5
    label: str = "motion_skin"  # honest label: skin-gated motion, still not a confirmed hand; see safety_zone HUMAN_LABELS
    confidence: float = 0.78
    stop_pose: Pose = field(default_factory=lambda: Pose(x=0.0, y=0.4, z=0.4))
    # Two HSV ranges cover wrap-around for warm/red skin tones; tuned for
    # typical East/South-East Asian indoor lighting and adjustable from YAML.
    hsv_lower_a: tuple[int, int, int] = (0, 30, 60)
    hsv_upper_a: tuple[int, int, int] = (20, 170, 255)
    hsv_lower_b: tuple[int, int, int] = (160, 30, 60)
    hsv_upper_b: tuple[int, int, int] = (179, 170, 255)
    skin_required: bool = True
    skin_overlap_ratio_min: float = 0.18

def _as_tuple3(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(x) for x in value)  # type: ignore[return-value]
    return default

class SkinMotionObjectDetector:
    """Detect hand-shaped foreground inside a configured ROI.

    A frame triggers when:

    1. Foreground motion area inside the ROI exceeds ``min_area_px``.
    2. (optional) Skin-tone pixels overlap the foreground mask by at least
       ``skin_overlap_ratio_min`` of the foreground area.

    This is a prototype, not a final hand detector. It is meant to be a more
    selective drop-in replacement for the pure ROI motion detector while we
    wait for a stable MediaPipe / YOLO integration.
    """

    def __init__(self, config: SkinMotionConfig | None = None):
        self.config = config or SkinMotionConfig()
        self._background: Any | None = None
        self._frames_seen = 0
        self.last_motion_area: float = 0.0
        self.last_skin_overlap_ratio: float = 0.0
        self.last_roi_box_px: tuple[int, int, int, int] | None = None
        self.last_triggered: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SkinMotionObjectDetector":
        data = data or {}
        roi = tuple(float(x) for x in data.get("roi", (0.30, 0.62, 0.70, 0.95)))
        pose_data = data.get("stop_pose", {})
        return cls(
            SkinMotionConfig(
                enabled=bool(data.get("enabled", False)),
                roi=roi,  # type: ignore[arg-type]
                min_area_px=float(data.get("min_area_px", 1500.0)),
                motion_threshold=int(data.get("motion_threshold", data.get("threshold", 28))),
                blur_kernel=int(data.get("blur_kernel", 5)),
                warmup_frames=int(data.get("warmup_frames", 5)),
                label=str(data.get("label", "motion_skin")),
                confidence=float(data.get("confidence", 0.78)),
                stop_pose=Pose(
                    x=float(pose_data.get("x", 0.0)),
                    y=float(pose_data.get("y", 0.4)),
                    z=float(pose_data.get("z", 0.4)),
                ),
                hsv_lower_a=_as_tuple3(data.get("hsv_lower_a"), (0, 30, 60)),
                hsv_upper_a=_as_tuple3(data.get("hsv_upper_a"), (20, 170, 255)),
                hsv_lower_b=_as_tuple3(data.get("hsv_lower_b"), (160, 30, 60)),
                hsv_upper_b=_as_tuple3(data.get("hsv_upper_b"), (179, 170, 255)),
                skin_required=bool(data.get("skin_required", True)),
                skin_overlap_ratio_min=float(data.get("skin_overlap_ratio_min", 0.18)),
            )
        )

    def reset(self) -> None:
        self._background = None
        self._frames_seen = 0
        self.last_motion_area = 0.0
        self.last_skin_overlap_ratio = 0.0
        self.last_triggered = False

    def detect(self, packet: CameraPacket) -> list[ObjectObservation]:
        if not self.config.enabled:
            self.last_triggered = False
            self.last_motion_area = 0.0
            self.last_skin_overlap_ratio = 0.0
            return []
        frame = packet.rgb if packet.rgb is not None else packet.mono
        if frame is None:
            self.last_triggered = False
            return []
        if len(frame.shape) == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            bgr = frame

        h, w = bgr.shape[:2]
        x0n, y0n, x1n, y1n = self.config.roi
        x0 = int(max(0.0, min(1.0, x0n)) * w)
        y0 = int(max(0.0, min(1.0, y0n)) * h)
        x1 = int(max(0.0, min(1.0, x1n)) * w)
        y1 = int(max(0.0, min(1.0, y1n)) * h)
        if x1 <= x0 or y1 <= y0:
            self.last_roi_box_px = None
            self.last_triggered = False
            return []
        self.last_roi_box_px = (x0, y0, x1, y1)
        roi = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        k = max(1, int(self.config.blur_kernel))
        if k % 2 == 0:
            k += 1
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        if self._background is None:
            self._background = gray.astype("float32")
            self._frames_seen = 1
            self.last_triggered = False
            self.last_motion_area = 0.0
            self.last_skin_overlap_ratio = 0.0
            return []

        self._frames_seen += 1
        cv2.accumulateWeighted(gray, self._background, 0.04)
        bg = cv2.convertScaleAbs(self._background)
        diff = cv2.absdiff(gray, bg)
        _, motion_mask = cv2.threshold(diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), np.uint8), iterations=1)
        motion_area = float(np.count_nonzero(motion_mask))
        self.last_motion_area = motion_area

        skin_overlap_ratio = 0.0
        if self.config.skin_required:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_a = cv2.inRange(hsv, np.array(self.config.hsv_lower_a, dtype=np.uint8), np.array(self.config.hsv_upper_a, dtype=np.uint8))
            mask_b = cv2.inRange(hsv, np.array(self.config.hsv_lower_b, dtype=np.uint8), np.array(self.config.hsv_upper_b, dtype=np.uint8))
            skin_mask = cv2.bitwise_or(mask_a, mask_b)
            inter = cv2.bitwise_and(motion_mask, skin_mask)
            inter_area = float(np.count_nonzero(inter))
            skin_overlap_ratio = (inter_area / motion_area) if motion_area > 0 else 0.0
        self.last_skin_overlap_ratio = skin_overlap_ratio

        if self._frames_seen <= self.config.warmup_frames:
            self.last_triggered = False
            return []

        motion_ok = motion_area >= self.config.min_area_px
        skin_ok = (not self.config.skin_required) or skin_overlap_ratio >= self.config.skin_overlap_ratio_min

        # Compute bounding box of largest motion contour for debug drawing
        bbox_px = None
        if motion_area > 0 and motion_ok and skin_ok:
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                xc, yc, wc, hc = cv2.boundingRect(largest)
                bbox_px = (x0 + xc, y0 + yc, x0 + xc + wc, y0 + yc + hc)

        self.last_triggered = bool(motion_ok and skin_ok)
        if not self.last_triggered:
            return []
        return [ObjectObservation(self.config.label, self.config.stop_pose, self.config.confidence, bbox_px=bbox_px)]

    def draw_debug(self, frame: Any) -> Any:
        if frame is None or self.last_roi_box_px is None:
            return frame
        x0, y0, x1, y1 = self.last_roi_box_px
        color = (0, 0, 255) if self.last_triggered else (0, 255, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            frame,
            f"skin_motion area={self.last_motion_area:.0f} "
            f"skin={self.last_skin_overlap_ratio:.2f} "
            f"trigger={self.last_triggered}",
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame
