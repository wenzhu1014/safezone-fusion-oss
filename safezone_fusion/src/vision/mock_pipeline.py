from __future__ import annotations

from typing import List

from ..models import MarkerObservation, ObjectObservation, Pose
from .interfaces import CameraPacket, PirPacket

def marker(mid=123, x=0.0, y=1.0, z=0.45, yaw=0.0, stable=6, speed=0.0, visible=True):
    return MarkerObservation(mid, Pose(x, y, z, yaw), stable, speed, visible)

def obj(label, x=0.0, y=1.0, z=0.4, confidence=1.0):
    return ObjectObservation(label, Pose(x, y, z), confidence)

class MockMarkerDetector:
    def __init__(self, observations: List[MarkerObservation | None]):
        self.observations = observations
        self.index = 0

    def detect(self, packet: CameraPacket) -> MarkerObservation | None:
        if self.index >= len(self.observations):
            return None
        out = self.observations[self.index]
        self.index += 1
        return out

class MockObjectDetector:
    def __init__(self, observations: List[list[ObjectObservation]]):
        self.observations = observations
        self.index = 0

    def detect(self, packet: CameraPacket) -> list[ObjectObservation]:
        if self.index >= len(self.observations):
            return []
        out = self.observations[self.index]
        self.index += 1
        return out

class MockPirReader:
    def __init__(self, signals: List[bool]):
        self.signals = signals
        self.index = 0

    def read(self) -> PirPacket:
        if self.index >= len(self.signals):
            return PirPacket(False, timestamp=float(self.index))
        triggered = self.signals[self.index]
        self.index += 1
        return PirPacket(triggered, timestamp=float(self.index))
