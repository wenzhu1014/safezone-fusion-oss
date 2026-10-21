from __future__ import annotations

from typing import Any, Dict

from .fusion import FusionDecisionEngine
from .human_access import HumanAccessController
from .interfaces import (
    ActuatorOutput,
    BusinessInput,
    DoorAction,
    MotorAction,
    OperationMode,
    RuntimeInput,
    RuntimeOutput,
)
from .models import Decision, FusionResult

class SafeZoneOrchestrator:
    """v0.3 orchestration layer.

    It keeps core algorithms sensor-agnostic and actuator-agnostic:
    - Business system provides OperationMode / SlotRequest / docking flag.
    - Vision layer provides SensorFrame.
    - Orchestrator returns abstract motor/door/voice actions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cart_engine = FusionDecisionEngine(config)
        self.human_access = HumanAccessController(config)

    def tick(self, inp: RuntimeInput) -> RuntimeOutput:
        mode = inp.business.mode

        if mode == OperationMode.CART_DOCKING:
            fusion = self.cart_engine.decide(inp.sensors, docking_requested=inp.business.docking_requested)
            return RuntimeOutput(mode, self._map_cart_decision(fusion), fusion)

        if mode == OperationMode.HUMAN_ACCESS:
            if inp.business.slot_request is None:
                fusion = FusionResult(Decision.IDLE, "human_access_missing_slot_request")
                return RuntimeOutput(mode, ActuatorOutput(reason=fusion.reason, raw_decision=fusion.decision.value), fusion)
            fusion = self.human_access.update(inp.business.slot_request, inp.sensors)
            return RuntimeOutput(mode, self._map_human_decision(fusion), fusion)

        fusion = FusionResult(Decision.IDLE, "runtime_idle")
        return RuntimeOutput(mode, ActuatorOutput(reason=fusion.reason, raw_decision=fusion.decision.value), fusion)

    def _map_cart_decision(self, fusion: FusionResult) -> ActuatorOutput:
        if fusion.decision == Decision.STOP:
            return ActuatorOutput(MotorAction.STOP, DoorAction.KEEP_CLOSED, fusion.voice, fusion.reason, fusion.decision.value)
        if fusion.decision == Decision.ALLOW_CART:
            return ActuatorOutput(MotorAction.ALLOW_CART_DOCKING, DoorAction.NONE, fusion.voice, fusion.reason, fusion.decision.value)
        if fusion.decision == Decision.WARNING:
            return ActuatorOutput(MotorAction.HOLD, DoorAction.NONE, fusion.voice, fusion.reason, fusion.decision.value)
        return ActuatorOutput(MotorAction.HOLD, DoorAction.NONE, fusion.voice, fusion.reason, fusion.decision.value)

    def _map_human_decision(self, fusion: FusionResult) -> ActuatorOutput:
        if fusion.decision == Decision.STOP:
            return ActuatorOutput(MotorAction.STOP, DoorAction.KEEP_CLOSED, fusion.voice, fusion.reason, fusion.decision.value)
        if fusion.decision == Decision.ASK_HUMAN_CLEAR:
            return ActuatorOutput(MotorAction.HOLD, DoorAction.KEEP_CLOSED, fusion.voice, fusion.reason, fusion.decision.value)
        if fusion.decision == Decision.MOVE_TO_PARK:
            return ActuatorOutput(MotorAction.MOVE_TO_PARK, DoorAction.KEEP_CLOSED, fusion.voice, fusion.reason, fusion.decision.value, fusion.debug)
        if fusion.decision == Decision.USER_ACCESS_ALLOWED:
            return ActuatorOutput(MotorAction.LOCK_STILL, DoorAction.OPEN_SLOT, fusion.voice, fusion.reason, fusion.decision.value)
        if fusion.decision == Decision.MOVE_TO_STANDBY:
            return ActuatorOutput(MotorAction.MOVE_TO_STANDBY, DoorAction.KEEP_CLOSED, fusion.voice, fusion.reason, fusion.decision.value)
        return ActuatorOutput(MotorAction.HOLD, DoorAction.NONE, fusion.voice, fusion.reason, fusion.decision.value)
