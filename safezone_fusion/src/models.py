from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class Decision(str, Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    STOP = "STOP"
    ALLOW_CART = "ALLOW_CART"
    ASK_HUMAN_CLEAR = "ASK_HUMAN_CLEAR"
    MOVE_TO_PARK = "MOVE_TO_PARK"
    USER_ACCESS_ALLOWED = "USER_ACCESS_ALLOWED"
    MOVE_TO_STANDBY = "MOVE_TO_STANDBY"

class StopReason(str, Enum):
    NONE = "none"
    HUMAN_IN_STOP_ZONE = "human_in_stop_zone"
    LIMB_IN_STOP_ZONE = "limb_in_stop_zone"
    UNKNOWN_OBSTACLE_IN_STOP_ZONE = "unknown_obstacle_in_stop_zone"
    PIR_WITH_UNKNOWN_OBSTACLE = "pir_with_unknown_obstacle"

@dataclass
class Pose:
    """Cabinet coordinate pose. x: left/right, y: height, z: distance from cabinet, meters."""

    x: float
    y: float
    z: float
    yaw_deg: float = 0.0

@dataclass
class MarkerObservation:
    marker_id: int
    pose: Pose
    stable_frames: int
    speed_mps: float = 0.0
    visible: bool = True
    smoothed_speed_mps: float = 0.0
    center_px: Optional[Tuple[float, float]] = None
    center_jitter_px: float = 0.0
    center_speed_px_s: float = 0.0
    calibrated_pose: bool = False
    coasting: bool = False  # True when re-emitted from last good marker during a transient detection miss

@dataclass
class ObjectObservation:
    label: str
    pose: Pose
    confidence: float = 1.0
    belongs_to_verified_cart: bool = False
    bbox_px: Optional[Tuple[int, int, int, int]] = None
    # image-space bounding box: (x0, y0, x1, y1) in pixels, for debug drawing

@dataclass
class SensorFrame:
    marker: Optional[MarkerObservation] = None
    objects: List[ObjectObservation] = field(default_factory=list)
    pir_triggered: bool = False
    timestamp: float = 0.0

@dataclass
class SlotRequest:
    """Human access request for a cabinet slot.

    slot_x/slot_y are the target slot center in cabinet 2D plane.
    head_x/head_y are current moving docking head center.
    """

    slot_id: str
    slot_x: float
    slot_y: float
    head_x: float
    head_y: float
    door_closed: bool = True
    user_done: bool = False

@dataclass
class CartAuthResult:
    verified: bool
    cart_id: Optional[str] = None
    marker_id: Optional[int] = None
    reason: str = ""
    pose: Optional[Pose] = None

@dataclass
class WaitingWindowResult:
    in_window: bool
    stable: bool
    reason: str = ""

@dataclass
class SafetyResult:
    stop: bool
    warning: bool
    reason: StopReason = StopReason.NONE
    objects_in_stop_zone: List[ObjectObservation] = field(default_factory=list)
    objects_in_warning_zone: List[ObjectObservation] = field(default_factory=list)

@dataclass
class FusionResult:
    decision: Decision
    reason: str
    voice: Optional[str] = None
    cart: Optional[CartAuthResult] = None
    waiting_window: Optional[WaitingWindowResult] = None
    safety: Optional[SafetyResult] = None
    debug: Dict[str, Any] = field(default_factory=dict)
