from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .models import FusionResult, SensorFrame, SlotRequest

class OperationMode(str, Enum):
    IDLE = "IDLE"
    CART_DOCKING = "CART_DOCKING"
    HUMAN_ACCESS = "HUMAN_ACCESS"

class MotorAction(str, Enum):
    HOLD = "HOLD"
    STOP = "STOP"
    ALLOW_CART_DOCKING = "ALLOW_CART_DOCKING"
    MOVE_TO_PARK = "MOVE_TO_PARK"
    MOVE_TO_STANDBY = "MOVE_TO_STANDBY"
    LOCK_STILL = "LOCK_STILL"

class DoorAction(str, Enum):
    NONE = "NONE"
    OPEN_SLOT = "OPEN_SLOT"
    KEEP_CLOSED = "KEEP_CLOSED"

@dataclass
class DevicePose:
    """Current pose of the moving docking device ("backpack") on the cabinet face.

    Supplied by the PLC/controller, which drives the motors and therefore knows
    exactly where the device is. Coordinates are in the cabinet-face frame; we
    keep them normalized (0..1 over the cabinet face) so the dynamic danger zone
    can be computed independently of physical units, and the pixel mapping is a
    separate calibration step.

    - x, y: device center on the cabinet face, normalized 0..1 (None = unknown)
    - moving: True when the device is actively driving (PLC knows this)
    - extension: optional reach/extension fraction if the mechanism telescopes
    """

    x: Optional[float] = None
    y: Optional[float] = None
    moving: bool = False
    extension: float = 0.0

    @property
    def known(self) -> bool:
        return self.x is not None and self.y is not None

@dataclass
class BusinessInput:
    """Input from the cabinet business system.

    - mode decides which flow is active.
    - slot_request exists in HUMAN_ACCESS mode.
    - docking_requested exists in CART_DOCKING mode.
    - device_pose carries the PLC-known position of the moving device, used to
      place the dynamic danger zone that follows the device across the face.
    """

    mode: OperationMode = OperationMode.IDLE
    slot_request: Optional[SlotRequest] = None
    docking_requested: bool = False
    device_pose: Optional[DevicePose] = None

@dataclass
class RuntimeInput:
    """Unified input for one control tick."""

    business: BusinessInput
    sensors: SensorFrame
    tick_id: int = 0

@dataclass
class ActuatorOutput:
    """Abstract output for motor/door/voice adapters."""

    motor: MotorAction = MotorAction.HOLD
    door: DoorAction = DoorAction.NONE
    voice: Optional[str] = None
    reason: str = ""
    raw_decision: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuntimeOutput:
    mode: OperationMode
    actuator: ActuatorOutput
    fusion: FusionResult
