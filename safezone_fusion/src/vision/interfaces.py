from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import MarkerObservation, ObjectObservation, SensorFrame

@dataclass
class CameraPacket:
    """One multi-camera/depth capture tick.

    Current PC prototype may provide rgb + nir/mono streams.
    Final cabinet deployment is expected to provide stereo/depth.
    Frames/depth maps/point clouds can be numpy arrays, SDK frames or paths.
    The interface keeps them as Any to avoid binding to OpenCV at this level.
    """

    rgb: Any = None
    nir: Any = None
    mono: Any = None
    left_rgb: Any = None
    right_rgb: Any = None
    left_depth: Any = None
    right_depth: Any = None
    depth: Any = None
    disparity: Any = None
    point_cloud: Any = None
    timestamp: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class PirPacket:
    triggered: bool = False
    timestamp: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

class MarkerDetector(Protocol):
    def detect(self, packet: CameraPacket) -> MarkerObservation | None:
        """Return the best current cart marker observation, or None."""

class ObjectDetector(Protocol):
    def detect(self, packet: CameraPacket) -> list[ObjectObservation]:
        """Return human/limb/unknown obstacle observations in cabinet coordinates."""

class PirReader(Protocol):
    def read(self) -> PirPacket:
        """Return current PIR/radar presence signal."""

class VisionSensorFusion:
    """Combines marker detector, object detector(s) and PIR into SensorFrame.

    ``object_detector`` may be a single detector or a list of detectors. When a
    list is given, every detector runs on each packet and their observations are
    merged into one ``SensorFrame.objects`` list. This enables a layered safety
    setup, e.g. mediapipe_hand (semantic, emits 'hand') running alongside
    roi_motion (motion safety-net, emits 'motion'): a bare forearm that the hand
    model misses is still caught by the motion detector. See safety_zone label
    tiers for how the merged labels are prioritised.
    """

    def __init__(self, marker_detector: MarkerDetector, object_detector, pir_reader: PirReader):
        self.marker_detector = marker_detector
        # Normalise to a list internally while keeping single-detector access.
        if isinstance(object_detector, (list, tuple)):
            self.object_detectors = list(object_detector)
        else:
            self.object_detectors = [object_detector]
        self.pir_reader = pir_reader

    @property
    def object_detector(self):
        """Back-compat accessor: the first (primary) object detector."""
        return self.object_detectors[0] if self.object_detectors else None

    def read(self, packet: CameraPacket) -> SensorFrame:
        marker = self.marker_detector.detect(packet)
        # Pass marker image-space bbox to downstream object detectors via packet.meta,
        # so motion-based detectors can mask out the marker itself and avoid mistaking
        # cart marker movement for a limb. Stays None when marker not visible / no corners.
        marker_bbox = self._marker_bbox_px()
        if marker_bbox is not None:
            packet.meta = dict(packet.meta or {})
            packet.meta["marker_bbox_px"] = marker_bbox
        objects: list[ObjectObservation] = []
        for det in self.object_detectors:
            objects.extend(det.detect(packet))
        pir = self.pir_reader.read()
        return SensorFrame(marker=marker, objects=objects, pir_triggered=pir.triggered, timestamp=packet.timestamp)

    def draw_debug(self, frame):
        """Run draw_debug on every detector that supports it (in order)."""
        for det in self.object_detectors:
            if hasattr(det, "draw_debug"):
                frame = det.draw_debug(frame)
        return frame

    def _marker_bbox_px(self) -> tuple[int, int, int, int] | None:
        """Return the most recent ArUco marker bbox in image pixels, or None.

        We rely on duck-typed access to ``last_corners`` on the marker detector.
        Any detector that exposes a numpy array of shape (N, 4, 2) under that
        name will work; otherwise we silently return None and downstream
        masking is disabled.
        """
        det = self.marker_detector
        corners = getattr(det, "last_corners", None)
        if corners is None:
            return None
        try:
            import numpy as np
            pts = np.asarray(corners[0]).reshape(-1, 2)
            xs, ys = pts[:, 0], pts[:, 1]
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                return None
            return (x0, y0, x1, y1)
        except Exception:
            return None
