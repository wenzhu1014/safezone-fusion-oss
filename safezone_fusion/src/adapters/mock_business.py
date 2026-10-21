from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..interfaces import BusinessInput, OperationMode
from ..models import SlotRequest

@dataclass
class BusinessEvent:
    mode: OperationMode
    docking_requested: bool = False
    slot_request: SlotRequest | None = None

class MockBusinessAdapter:
    """Scripted business-system adapter for demos.

    Real implementation should read from cabinet backend / UI / order system.
    """

    def __init__(self, events: List[BusinessEvent]):
        self.events = events
        self.index = 0

    def read(self) -> BusinessInput:
        if self.index >= len(self.events):
            return BusinessInput(mode=OperationMode.IDLE)
        event = self.events[self.index]
        self.index += 1
        return BusinessInput(
            mode=event.mode,
            docking_requested=event.docking_requested,
            slot_request=event.slot_request,
        )
