from __future__ import annotations

from enum import Enum
from math import hypot
from typing import Any, Dict

from .models import Decision, FusionResult, SensorFrame, SlotRequest
from .safety_zone import SafetyZoneMonitor

class HumanAccessState(str, Enum):
    IDLE = "IDLE"
    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    ASK_USER_CLEAR = "ASK_USER_CLEAR"
    MOVE_TO_PARK = "MOVE_TO_PARK"
    USER_ACCESS_ALLOWED = "USER_ACCESS_ALLOWED"
    ASK_CLEAR_FOR_RETURN = "ASK_CLEAR_FOR_RETURN"
    MOVE_TO_STANDBY = "MOVE_TO_STANDBY"
    DONE = "DONE"
    STOPPED = "STOPPED"

class HumanAccessController:
    """v0.2 artificial human access flow.

    Principle:
    - Device never moves while human/limb is in stop_zone.
    - After device moves away and locks still, user may enter the access area.
    - Before returning, ask user to clear again.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.human_cfg = config.get("human_access", {})
        self.voice = config.get("voice", {})
        self.safety = SafetyZoneMonitor(config)
        self.state = HumanAccessState.IDLE
        self._slot_blocked = False

    def is_slot_blocked(self, req: SlotRequest) -> bool:
        radius = float(self.human_cfg.get("slot_block_radius_m", 0.12))
        return hypot(req.slot_x - req.head_x, req.slot_y - req.head_y) <= radius

    def _motion_area_clear(self, frame: SensorFrame) -> bool:
        safety = self.safety.check(frame, cart_verified=False)
        if safety.stop:
            return False
        if self.human_cfg.get("require_warning_zone_clear_before_move", False) and safety.warning:
            return False
        return True

    def update(self, req: SlotRequest, frame: SensorFrame) -> FusionResult:
        safety = self.safety.check(frame, cart_verified=False)

        if self.state == HumanAccessState.IDLE:
            self.state = HumanAccessState.REQUEST_RECEIVED
            self._slot_blocked = self.is_slot_blocked(req)

        if self.state == HumanAccessState.REQUEST_RECEIVED:
            if not self._slot_blocked:
                self.state = HumanAccessState.USER_ACCESS_ALLOWED
                return FusionResult(
                    Decision.USER_ACCESS_ALLOWED,
                    "slot_not_blocked_device_locked",
                    voice=self.voice.get("human_access_allowed"),
                    safety=safety,
                )
            self.state = HumanAccessState.ASK_USER_CLEAR

        if self.state == HumanAccessState.ASK_USER_CLEAR:
            if not self._motion_area_clear(frame):
                return FusionResult(
                    Decision.ASK_HUMAN_CLEAR,
                    "slot_blocked_waiting_human_clear_before_move",
                    voice=self.voice.get("human_step_back"),
                    safety=safety,
                )
            self.state = HumanAccessState.MOVE_TO_PARK
            return FusionResult(
                Decision.MOVE_TO_PARK,
                "safety_clear_move_to_park_position",
                voice=self.voice.get("human_move_away"),
                safety=safety,
                debug={"park_position": self.human_cfg.get("park_position")},
            )

        if self.state == HumanAccessState.MOVE_TO_PARK:
            # In real deployment this state waits for motor feedback. In v0.2 demo one update means motion complete.
            if safety.stop:
                self.state = HumanAccessState.STOPPED
                return FusionResult(Decision.STOP, safety.reason.value, voice=self.voice.get("stop_human"), safety=safety)
            self.state = HumanAccessState.USER_ACCESS_ALLOWED
            return FusionResult(
                Decision.USER_ACCESS_ALLOWED,
                "device_parked_and_locked_user_access_allowed",
                voice=self.voice.get("human_access_allowed"),
                safety=safety,
            )

        if self.state == HumanAccessState.USER_ACCESS_ALLOWED:
            if not req.user_done or not req.door_closed:
                return FusionResult(
                    Decision.USER_ACCESS_ALLOWED,
                    "waiting_user_finish_with_device_locked",
                    voice=None,
                    safety=safety,
                )
            self.state = HumanAccessState.ASK_CLEAR_FOR_RETURN

        if self.state == HumanAccessState.ASK_CLEAR_FOR_RETURN:
            if not self._motion_area_clear(frame):
                return FusionResult(
                    Decision.ASK_HUMAN_CLEAR,
                    "waiting_human_clear_before_return",
                    voice=self.voice.get("human_return_step_back"),
                    safety=safety,
                )
            self.state = HumanAccessState.MOVE_TO_STANDBY
            return FusionResult(
                Decision.MOVE_TO_STANDBY,
                "safety_clear_return_to_standby",
                voice=self.voice.get("human_move_away"),
                safety=safety,
            )

        if self.state == HumanAccessState.MOVE_TO_STANDBY:
            if safety.stop:
                self.state = HumanAccessState.STOPPED
                return FusionResult(Decision.STOP, safety.reason.value, voice=self.voice.get("stop_human"), safety=safety)
            self.state = HumanAccessState.DONE
            return FusionResult(Decision.IDLE, "human_access_flow_done", safety=safety)

        if self.state == HumanAccessState.STOPPED:
            if not safety.stop:
                self.state = HumanAccessState.ASK_USER_CLEAR
            return FusionResult(Decision.STOP, "stopped_waiting_clear", voice=self.voice.get("stop_human"), safety=safety)

        return FusionResult(Decision.IDLE, "human_access_idle", safety=safety)
