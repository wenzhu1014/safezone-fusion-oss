from __future__ import annotations

from typing import Any, Dict, Optional

from .geometry import in_box, in_image_corridor, in_waiting_window, near_marker_region
from .models import MarkerObservation, ObjectObservation, SafetyResult, SensorFrame, StopReason

HUMAN_LABELS = {"person", "human", "hand", "arm", "leg", "foot", "head", "body", "limb"}
# Label-tier semantics (semantic path, since 2026-06-22):
#   - person/human            -> HUMAN_IN_STOP_ZONE   (highest)
#   - hand/arm/limb/...        -> LIMB_IN_STOP_ZONE    (confirmed human limb)
#   - anything else (e.g.      -> UNKNOWN_OBSTACLE_IN_STOP_ZONE (still STOPs)
#     'motion', 'motion_skin')
# Only semantic detectors that can actually confirm a hand (mediapipe_hand) emit
# 'hand'. Motion-only detectors (roi_motion/skin_motion) emit 'motion'/'motion_skin'
# so they STILL trigger a STOP via the UNKNOWN_OBSTACLE path, but never falsely
# claim a human limb. This keeps the motion safety-net while making labels honest.

class SafetyZoneMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_zone = config["safety"]["stop_zone"]
        self.warning_zone = config["safety"]["warning_zone"]
        self.verified_cart_region = config.get("verified_cart_region", {})
        self.docking_corridor = config.get("docking_corridor", config.get("waiting_window", {}))
        self.pir_stop_when_unknown = bool(config["safety"].get("pir_stop_when_unknown_in_stop_zone", True))
        # Corridor mode: "pose" (3D x/z/yaw, default) or "image" (normalized
        # pixel region for a fixed overhead camera). Image mode does not depend
        # on fragile depth/yaw estimation. Frame size needed for image mode.
        self.corridor_mode = str(self.docking_corridor.get("mode", "pose")).lower() if self.docking_corridor else "pose"
        self.frame_w = int(config.get("camera", {}).get("width", 0)) if isinstance(config.get("camera"), dict) else 0
        self.frame_h = int(config.get("camera", {}).get("height", 0)) if isinstance(config.get("camera"), dict) else 0

    def _marker_in_corridor(self, frame: SensorFrame) -> bool:
        """Whether the marker is inside the docking corridor, honoring the mode."""
        if not self.docking_corridor:
            return True  # no corridor configured -> do not block on corridor
        if self.corridor_mode == "image":
            fw = self.frame_w or getattr(frame, "frame_w", 0) or 0
            fh = self.frame_h or getattr(frame, "frame_h", 0) or 0
            return in_image_corridor(getattr(frame.marker, "center_px", None),
                                     self.docking_corridor, fw, fh)
        return in_waiting_window(frame.marker.pose, self.docking_corridor)

    def mark_cart_region(self, frame: SensorFrame, cart_verified: bool) -> None:
        if not cart_verified or frame.marker is None or not self.verified_cart_region:
            return

        # 安全豁免必须同时满足：
        # 1) 小车 marker 已授权；
        # 2) marker 处于接驳走廊/等待窗口（pose 或 image 模式）；
        # 3) 物体点云在 marker 附近的“小车本体区域”；
        # 人体/肢体永远不在这里豁免，后续 stop 判定仍最高优先级。
        if self.docking_corridor and not self._marker_in_corridor(frame):
            return

        for obj in frame.objects:
            if obj.label.lower() in HUMAN_LABELS:
                continue
            if near_marker_region(obj.pose, frame.marker.pose, self.verified_cart_region):
                obj.belongs_to_verified_cart = True

    def check(self, frame: SensorFrame, cart_verified: bool = False) -> SafetyResult:
        self.mark_cart_region(frame, cart_verified)

        objects_in_stop = []
        objects_in_warning = []

        for obj in frame.objects:
            if in_box(obj.pose, self.stop_zone):
                objects_in_stop.append(obj)
            elif in_box(obj.pose, self.warning_zone):
                objects_in_warning.append(obj)

        for obj in objects_in_stop:
            label = obj.label.lower()
            if label in HUMAN_LABELS:
                if label == "person" or label == "human":
                    return SafetyResult(True, True, StopReason.HUMAN_IN_STOP_ZONE, objects_in_stop, objects_in_warning)
                return SafetyResult(True, True, StopReason.LIMB_IN_STOP_ZONE, objects_in_stop, objects_in_warning)

        unknown_in_stop = [obj for obj in objects_in_stop if not obj.belongs_to_verified_cart]
        if unknown_in_stop:
            if frame.pir_triggered and self.pir_stop_when_unknown:
                return SafetyResult(True, True, StopReason.PIR_WITH_UNKNOWN_OBSTACLE, objects_in_stop, objects_in_warning)
            return SafetyResult(True, True, StopReason.UNKNOWN_OBSTACLE_IN_STOP_ZONE, objects_in_stop, objects_in_warning)

        warning = bool(objects_in_warning)
        return SafetyResult(False, warning, StopReason.NONE, objects_in_stop, objects_in_warning)
