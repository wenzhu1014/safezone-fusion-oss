"""MediaPipe Hands object detector for mediapipe 0.10+ tasks API.

Draws skeleton manually with OpenCV circles/lines — no dependency on
drawing_utils which is missing from the Windows pypi wheel.

Uses an explicitly supplied local model file; no automatic model download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from pathlib import Path

import cv2
import numpy as np

from ..models import ObjectObservation, Pose
from .interfaces import CameraPacket

# Hand skeleton connections: pairs of landmark indices
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                    # palm
]

@dataclass
class MediaPipeHandConfig:
    enabled: bool = False
    model_path: str | None = None
    roi: tuple[float, float, float, float] = (0.30, 0.62, 0.70, 0.95)
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    label: str = "hand"
    confidence: float = 0.75
    stop_pose: Pose = field(default_factory=lambda: Pose(x=0.0, y=0.4, z=0.4))

class MediaPipeHandObjectDetector:
    """MediaPipe Hands detection producing full hand skeleton.

    Every detected hand gets:
    - purple circles at each of the 21 landmarks (wrist, knuckles, finger tips)
    - green lines for connections forming the "火柴人" skeleton
    - green bounding box with label
    """

    def __init__(self, config: MediaPipeHandConfig | None = None):
        self.config = config or MediaPipeHandConfig()
        self._initialized = False
        self._hand_landmarker = None
        self.last_landmarks_list: List[Any] = []
        self.last_bboxes: list[tuple[int, int, int, int]] = []
        self.last_triggered: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MediaPipeHandObjectDetector":
        data = data or {}
        roi = data.get("roi", (0.30, 0.62, 0.70, 0.95))
        pose_data = data.get("stop_pose", {})
        return cls(
            MediaPipeHandConfig(
                enabled=bool(data.get("enabled", False)),
                model_path=data.get("model_path"),
                roi=tuple(float(x) for x in roi),
                num_hands=int(data.get("num_hands", 2)),
                min_detection_confidence=float(data.get("min_detection_confidence", 0.5)),
                min_hand_presence_confidence=float(data.get("min_hand_presence_confidence", 0.5)),
                min_tracking_confidence=float(data.get("min_tracking_confidence", 0.5)),
                label=str(data.get("label", "hand")),
                confidence=float(data.get("confidence", 0.75)),
                stop_pose=Pose(
                    x=float(pose_data.get("x", 0.0)),
                    y=float(pose_data.get("y", 0.4)),
                    z=float(pose_data.get("z", 0.4)),
                ),
            )
        )

    def _lazy_init(self):
        if self._initialized:
            return
        if not self.config.model_path:
            raise RuntimeError("mediapipe_hand requires an explicit local model_path")
        model_path = Path(self.config.model_path).expanduser()
        if not model_path.is_file():
            raise RuntimeError("mediapipe_hand model_path must point to an existing local file")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        except Exception as exc:
            raise RuntimeError(
                "mediapipe_hand requires a compatible optional mediapipe installation"
            ) from exc

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(options)
        self._initialized = True
        self.last_landmarks_list = []
        self.last_bboxes = []

    def reset(self) -> None:
        self._initialized = False
        self._hand_landmarker = None
        self.last_landmarks_list = []
        self.last_bboxes = []
        self.last_triggered = False

    def detect(self, packet: CameraPacket) -> list[ObjectObservation]:
        self.last_landmarks_list = []
        self.last_bboxes = []
        self.last_triggered = False
        if not self.config.enabled:
            return []
        frame = packet.rgb if packet.rgb is not None else packet.mono
        if frame is None:
            return []
        self._lazy_init()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self._hand_landmarker.detect(mp_image)
        if not results.hand_landmarks:
            return []
        objects = []
        h, w = img_rgb.shape[:2]
        for hand_landmarks in results.hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks]
            y_coords = [lm.y * h for lm in hand_landmarks]
            x0 = int(max(0, min(x_coords)))
            x1 = int(min(w, max(x_coords)))
            y0 = int(max(0, min(y_coords)))
            y1 = int(min(h, max(y_coords)))
            bbox = (x0, y0, x1, y1)
            self.last_landmarks_list.append(hand_landmarks)
            self.last_bboxes.append(bbox)
            objects.append(ObjectObservation(
                self.config.label,
                self.config.stop_pose,
                self.config.confidence,
                bbox_px=bbox,
            ))
        self.last_triggered = len(objects) > 0
        return objects

    def draw_debug(self, frame: Any) -> Any:
        if frame is None or not self._initialized or not self.last_landmarks_list:
            return frame
        h, w = frame.shape[:2]
        for landmarks in self.last_landmarks_list:
            # Collect pixel positions of all 21 landmarks
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            # Draw green skeleton connections first
            for a, b in _HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
            # Draw purple circles at joints
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (255, 0, 255), -1, cv2.LINE_AA)
        # Draw green bounding box
        for (x0, y0, x1, y1) in self.last_bboxes:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                frame, self.config.label,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
        return frame
