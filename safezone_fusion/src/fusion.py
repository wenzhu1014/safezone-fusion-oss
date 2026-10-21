from __future__ import annotations

from typing import Any, Dict

from .cart_auth import CartAuthenticator
from .models import Decision, FusionResult, SensorFrame
from .safety_zone import SafetyZoneMonitor
from .waiting_window import WaitingWindowChecker

class FusionDecisionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cart_auth = CartAuthenticator(config)
        self.waiting_window = WaitingWindowChecker(config)
        self.safety = SafetyZoneMonitor(config)
        self.voice = config.get("voice", {})

    def decide(self, frame: SensorFrame, docking_requested: bool = True) -> FusionResult:
        cart = self.cart_auth.verify(frame.marker)
        window = self.waiting_window.check(frame.marker)
        safety = self.safety.check(frame, cart_verified=cart.verified)

        if safety.stop:
            if "human" in safety.reason.value or "limb" in safety.reason.value:
                voice = self.voice.get("stop_human")
            else:
                voice = self.voice.get("stop_obstacle")
            return FusionResult(
                Decision.STOP,
                safety.reason.value,
                voice=voice,
                cart=cart,
                waiting_window=window,
                safety=safety,
            )

        if safety.warning:
            return FusionResult(
                Decision.WARNING,
                "object_in_warning_zone",
                voice=self.voice.get("warning"),
                cart=cart,
                waiting_window=window,
                safety=safety,
            )

        if not docking_requested:
            return FusionResult(Decision.IDLE, "docking_not_requested", cart=cart, waiting_window=window, safety=safety)

        if not cart.verified:
            return FusionResult(
                Decision.IDLE,
                f"cart_not_verified:{cart.reason}",
                voice=self.voice.get("wait_cart"),
                cart=cart,
                waiting_window=window,
                safety=safety,
            )

        if not window.in_window or not window.stable:
            return FusionResult(
                Decision.WARNING,
                window.reason,
                voice=self.voice.get("wait_cart"),
                cart=cart,
                waiting_window=window,
                safety=safety,
            )

        return FusionResult(
            Decision.ALLOW_CART,
            "authorized_cart_in_waiting_window_and_safety_clear",
            voice=self.voice.get("allow_cart"),
            cart=cart,
            waiting_window=window,
            safety=safety,
        )
