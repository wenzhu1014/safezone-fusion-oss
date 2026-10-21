from __future__ import annotations

from typing import List

from ..interfaces import ActuatorOutput

class MockActuatorAdapter:
    """Collects actuator outputs for demo and test.

    Real implementation should map MotorAction / DoorAction / voice text
    to motor controller, door lock controller and speaker/TTS.
    """

    def __init__(self):
        self.history: List[ActuatorOutput] = []

    def apply(self, output: ActuatorOutput) -> None:
        self.history.append(output)
        print(
            f"motor={output.motor.value:<20} door={output.door.value:<12} "
            f"decision={str(output.raw_decision):<20} reason={output.reason}"
        )
        if output.voice:
            print(f"  voice: {output.voice}")
        if output.debug:
            print(f"  debug: {output.debug}")
