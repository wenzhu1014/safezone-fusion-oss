from __future__ import annotations

from typing import List

from ..models import MarkerObservation, ObjectObservation, Pose, SensorFrame

def marker(mid=123, x=0.0, y=1.0, z=0.45, yaw=0.0, stable=6, speed=0.0, visible=True):
    return MarkerObservation(mid, Pose(x, y, z, yaw), stable, speed, visible)

def obj(label, x=0.0, y=1.0, z=0.4, confidence=1.0):
    return ObjectObservation(label, Pose(x, y, z), confidence)

class MockSensorAdapter:
    """Scripted sensor adapter for demos.

    Real implementation should merge camera marker detection, stereo depth,
    human/limb detector, PIR/millimeter-wave radar into SensorFrame.
    """

    def __init__(self, frames: List[SensorFrame]):
        self.frames = frames
        self.index = 0

    def read(self) -> SensorFrame:
        if self.index >= len(self.frames):
            return SensorFrame(timestamp=float(self.index))
        frame = self.frames[self.index]
        frame.timestamp = float(self.index)
        self.index += 1
        return frame
