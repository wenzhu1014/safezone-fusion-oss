from __future__ import annotations

from enum import Enum

from .models import Decision, FusionResult

class SystemState(str, Enum):
    IDLE = "IDLE"
    SCAN_CART = "SCAN_CART"
    CART_ALLOWED = "CART_ALLOWED"
    WARNING = "WARNING"
    STOPPED = "STOPPED"
    CLEARING = "CLEARING"

class SafetyStateMachine:
    def __init__(self):
        self.state = SystemState.IDLE

    def update(self, result: FusionResult) -> SystemState:
        if result.decision == Decision.STOP:
            self.state = SystemState.STOPPED
        elif result.decision == Decision.ALLOW_CART:
            self.state = SystemState.CART_ALLOWED
        elif result.decision == Decision.WARNING:
            self.state = SystemState.WARNING
        elif result.decision == Decision.IDLE:
            if self.state == SystemState.STOPPED:
                self.state = SystemState.CLEARING
            else:
                self.state = SystemState.SCAN_CART
        return self.state
