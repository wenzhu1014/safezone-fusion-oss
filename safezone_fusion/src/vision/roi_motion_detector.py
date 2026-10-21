from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..models import ObjectObservation, Pose
from .interfaces import CameraPacket
from .illumination import IlluminationConfig, compensated_diff, clean_mask

@dataclass
class RoiMotionConfig:
    enabled: bool = False
    roi: tuple[float, float, float, float] = (0.35, 0.35, 0.65, 0.85)
    min_area_px: float = 400.0
    threshold: int = 28
    blur_kernel: int = 5
    warmup_frames: int = 5
    min_consecutive: int = 2
    label: str = "motion"  # honest label: motion-only detector cannot confirm a hand; see safety_zone HUMAN_LABELS
    confidence: float = 0.75
    stop_pose: Pose = field(default_factory=lambda: Pose(x=0.0, y=0.4, z=0.4))
    # Mask out the cart marker (and a small padding around it) before computing motion.
    # This prevents marker self-movement from being classified as a limb in the stop zone.
    # Disable by setting marker_mask_pad_px < 0.
    marker_mask_pad_px: int = 30
    # Padding (px) around the device-body mask before excluding it, so blur/edge
    # bleed from the moving device body is not detected just outside the mask.
    device_mask_pad_px: int = 12
    # Safety-critical (2026-06-22): hold a STATIC occupant. A motion-only detector
    # absorbs a stopped object into the background and the object silently
    # 'disappears', wrongly clearing the zone. When True, background learning is
    # frozen while the ROI is occupied, so a thing placed and left in the zone
    # keeps being detected until it is physically removed. A safety zone is judged
    # by occupancy (is something there) not motion (is it moving).
    hold_static_occupant: bool = True
    # Occupancy hysteresis + STOP latch (2026-06-22, fixes "object fully inside the
    # ROI makes STOP flicker"). A flat/large object has little internal texture,
    # so its diff area sits near min_area_px and a single frame dipping below the
    # threshold used to UNFREEZE the background -> avalanche absorption -> flicker.
    # Fix: two thresholds (enter high, stay low) + require N consecutive CLEAR
    # frames before releasing the latch. While latched, the background is never
    # learned, so a textureless object cannot erode the model. Release is slow
    # (safety-biased: prefer to keep stopping).
    occupied_keep_area_px: float = 150.0   # stay-occupied threshold (< min_area_px = enter threshold)
    clear_frames_to_release: int = 8       # consecutive clear frames required to drop the STOP latch

class RoiMotionObjectDetector:
    """Lightweight single-camera motion detector for early hand/obstacle testing.

    This is not a replacement for YOLO/depth. It is a controllable prototype that
    turns motion inside a configured image ROI into one `ObjectObservation`, so
    the existing SafetyZoneMonitor can be tested with real camera clips.

    ROI is normalized `(x_min, y_min, x_max, y_max)` in image coordinates. The
    output pose defaults inside the configured stop zone, because this detector
    represents "motion observed in the dangerous screen region" rather than a
    calibrated metric 3D position.
    """

    def __init__(self, config: RoiMotionConfig | None = None):
        self.config = config or RoiMotionConfig()
        self._background: Any | None = None
        self._frames_seen = 0
        self._consecutive_motion_frames = 0
        self.last_motion_area: float = 0.0
        self.last_roi_box_px: tuple[int, int, int, int] | None = None
        self.last_triggered: bool = False
        self.last_marker_mask_px: tuple[int, int, int, int] | None = None
        # Occupancy state machine
        self._occupied_latched: bool = False   # currently holding the zone occupied (STOP latched)
        self._clear_streak: int = 0             # consecutive frames the ROI looked clear
        # Optional dynamic overrides set per-frame by the orchestrator/demo from
        # the device-following danger zone (see following_zone.py). When set,
        # they take precedence over the static config.roi for THIS frame:
        #   _dynamic_roi:      normalized (x0,y0,x1,y1) danger box to watch
        #   _device_mask_px:   pixel (x0,y0,x1,y1) of the device body to exclude
        self._dynamic_roi: tuple[float, float, float, float] | None = None
        self._device_mask_px: tuple[int, int, int, int] | None = None
        self.last_device_mask_px: tuple[int, int, int, int] | None = None
        # Co-motion exemption (博's idea, 2026-06-22): the cart body moves
        # together with its authorized marker; an intruding hand does not. We
        # track the marker centroid velocity and the dominant motion-blob
        # velocity, and exempt the blob when it moves with an authorized,
        # in-corridor, MOVING marker. The semantic hand layer stays independent
        # and is never exempted, so layered defense is preserved.
        from ..comotion import MotionTracker, CoMotionConfig
        self._marker_tracker = MotionTracker()
        self._blob_tracker = MotionTracker()
        self._comotion_cfg = CoMotionConfig()
        self._marker_centroid_px: tuple[float, float] | None = None
        self._marker_exempt_active: bool = False  # authorized + in corridor
        self.last_comotion_reason: str = ""
        # Illumination robustness + shadow suppression (indoor & outdoor).
        self._illum_cfg = IlluminationConfig()

    def set_dynamic_roi(self, roi_norm: tuple[float, float, float, float] | None) -> None:
        """Set the danger-zone ROI for the next frame (normalized 0..1).

        Pass the danger_box from the following-zone so the watched region tracks
        the moving device instead of a static box. Pass None to revert to the
        static config.roi. Reset background when the ROI changes a lot so a new
        region is not falsely flagged from a stale background.
        """
        if roi_norm is None:
            if self._dynamic_roi is not None:
                self._dynamic_roi = None
                self._background = None
                self._frames_seen = 0
            return
        if self._dynamic_roi is None or self._roi_shifted(self._dynamic_roi, roi_norm):
            # ROI moved meaningfully -> background no longer valid for new pixels
            self._background = None
            self._frames_seen = 0
        self._dynamic_roi = roi_norm

    def set_device_mask_px(self, box_px: tuple[int, int, int, int] | None) -> None:
        """Set the device-body pixel box to EXCLUDE from obstacle detection.

        The device ("backpack") is not an obstacle; mask it out so its own
        movement never triggers STOP. Same mechanism as the marker mask.
        """
        self._device_mask_px = box_px

    def set_marker_comotion(self, centroid_px: tuple[float, float] | None,
                            exempt_active: bool) -> None:
        """Feed the marker centroid + whether co-motion exemption may apply.

        ``centroid_px``: marker center in pixels (None when marker not seen).
        ``exempt_active``: True only when the marker is authorized AND inside the
        docking corridor (the caller decides this). Co-motion exemption is
        allowed only when this is True; motion correlation alone never exempts.
        """
        self._marker_centroid_px = centroid_px
        self._marker_exempt_active = bool(exempt_active)

    @staticmethod
    def _roi_shifted(a, b, thresh: float = 0.04) -> bool:
        return any(abs(x - y) > thresh for x, y in zip(a, b))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoiMotionObjectDetector":
        data = data or {}
        roi = data.get("roi", [0.35, 0.35, 0.65, 0.85])
        pose_data = data.get("stop_pose", {})
        instance = cls(
            RoiMotionConfig(
                enabled=bool(data.get("enabled", False)),
                roi=tuple(float(x) for x in roi),
                min_area_px=float(data.get("min_area_px", 1200.0)),
                threshold=int(data.get("threshold", 28)),
                blur_kernel=int(data.get("blur_kernel", 5)),
                warmup_frames=int(data.get("warmup_frames", 5)),
                label=str(data.get("label", "motion")),
                confidence=float(data.get("confidence", 0.75)),
                stop_pose=Pose(
                    x=float(pose_data.get("x", 0.0)),
                    y=float(pose_data.get("y", 0.4)),
                    z=float(pose_data.get("z", 0.4)),
                ),
                min_consecutive=int(data.get("min_consecutive", 3)),
                marker_mask_pad_px=int(data.get("marker_mask_pad_px", 30)),
                hold_static_occupant=bool(data.get("hold_static_occupant", True)),
                occupied_keep_area_px=float(data.get("occupied_keep_area_px", 150.0)),
                clear_frames_to_release=int(data.get("clear_frames_to_release", 8)),
            )
        )
        instance._illum_cfg = IlluminationConfig.from_dict(data.get("illumination"))
        return instance

    def reset(self) -> None:
        self._background = None
        self._frames_seen = 0
        self._consecutive_motion_frames = 0
        self.last_motion_area = 0.0
        self.last_triggered = False
        self._occupied_latched = False
        self._clear_streak = 0

    def detect(self, packet: CameraPacket) -> list[ObjectObservation]:
        if not self.config.enabled:
            self.last_triggered = False
            self.last_motion_area = 0.0
            self._consecutive_motion_frames = 0
            return []
        frame = packet.rgb if packet.rgb is not None else packet.mono
        if frame is None:
            self.last_triggered = False
            self.last_motion_area = 0.0
            self._consecutive_motion_frames = 0
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        k = max(1, int(self.config.blur_kernel))
        if k % 2 == 0:
            k += 1
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        h, w = gray.shape[:2]
        x0n, y0n, x1n, y1n = self._dynamic_roi if self._dynamic_roi is not None else self.config.roi
        x0 = int(max(0.0, min(1.0, x0n)) * w)
        y0 = int(max(0.0, min(1.0, y0n)) * h)
        x1 = int(max(0.0, min(1.0, x1n)) * w)
        y1 = int(max(0.0, min(1.0, y1n)) * h)
        if x1 <= x0 or y1 <= y0:
            self.last_roi_box_px = None
            self.last_triggered = False
            self.last_motion_area = 0.0
            self._consecutive_motion_frames = 0
            return []
        self.last_roi_box_px = (x0, y0, x1, y1)
        roi = gray[y0:y1, x0:x1]

        if self._background is None:
            self._background = roi.astype("float32")
            self._frames_seen = 1
            self._consecutive_motion_frames = 0
            self.last_triggered = False
            self.last_motion_area = 0.0
            return []

        self._frames_seen += 1
        # NOTE: background adaptation is applied LATER and SELECTIVELY (see below),
        # not blindly every frame, so a static occupant is not absorbed.
        bg = cv2.convertScaleAbs(self._background)
        # Illumination-robust differencing: compensate a uniform brightness shift
        # (cloud/light change) so only LOCAL changes survive, then threshold.
        diff_f = compensated_diff(roi, bg, self._illum_cfg)
        diff = cv2.convertScaleAbs(diff_f)
        _, mask = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)
        # Shadow suppression: drop pixels that are a uniform darkening (cast
        # shadow), keeping real objects that change texture/occlude.
        mask = clean_mask(mask, roi, bg, self._illum_cfg)

        # Mask out the cart marker so its own movement is not classified as a limb.
        # The fusion layer writes the marker's image-space bbox into packet.meta.
        marker_bbox = None
        if isinstance(packet.meta, dict):
            marker_bbox = packet.meta.get("marker_bbox_px")
        self.last_marker_mask_px = None
        if marker_bbox and self.config.marker_mask_pad_px >= 0:
            mx0, my0, mx1, my1 = marker_bbox
            pad = int(self.config.marker_mask_pad_px)
            # Convert to ROI-local coordinates and clamp to the ROI rectangle
            lx0 = max(0, mx0 - pad - x0)
            ly0 = max(0, my0 - pad - y0)
            lx1 = min(x1 - x0, mx1 + pad - x0)
            ly1 = min(y1 - y0, my1 + pad - y0)
            if lx1 > lx0 and ly1 > ly0:
                mask[ly0:ly1, lx0:lx1] = 0
                self.last_marker_mask_px = (
                    max(x0, mx0 - pad),
                    max(y0, my0 - pad),
                    min(x1, mx1 + pad),
                    min(y1, my1 + pad),
                )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))        # Mask out the moving device body ("backpack") so its own movement across
        # the cabinet face is never classified as an obstacle. The device pose
        # comes from the PLC; the orchestrator converts it to a pixel box via
        # the following-zone + calibration and sets it with set_device_mask_px().
        self.last_device_mask_px = None
        if self._device_mask_px is not None:
            pad = max(0, int(self.config.device_mask_pad_px))
            dmx0, dmy0, dmx1, dmy1 = self._device_mask_px
            dmx0 -= pad; dmy0 -= pad; dmx1 += pad; dmy1 += pad
            lx0 = max(0, dmx0 - x0)
            ly0 = max(0, dmy0 - y0)
            lx1 = min(x1 - x0, dmx1 - x0)
            ly1 = min(y1 - y0, dmy1 - y0)
            if lx1 > lx0 and ly1 > ly0:
                mask[ly0:ly1, lx0:lx1] = 0
                self.last_device_mask_px = (
                    max(x0, dmx0), max(y0, dmy0), min(x1, dmx1), min(y1, dmy1)
                )

        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float(sum(cv2.contourArea(c) for c in contours))
        self.last_motion_area = area

        # Compute bounding box of largest contour for debug drawing
        bbox_px = None
        blob_centroid = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            xc, yc, wc, hc = cv2.boundingRect(largest)
            bbox_px = (x0 + xc, y0 + yc, x0 + xc + wc, y0 + yc + hc)
            blob_centroid = (x0 + xc + wc / 2.0, y0 + yc + hc / 2.0)

        # --- Co-motion exemption (博's idea): is the dominant motion blob the
        # cart body moving together with the authorized marker? Track marker and
        # blob velocities; exempt only when the marker is authorized+in-corridor
        # (exempt_active) AND moving AND the blob moves with it. ---
        self.last_comotion_reason = ""
        marker_vel = self._marker_tracker.update(self._marker_centroid_px)
        blob_vel = self._blob_tracker.update(blob_centroid)
        comotion_exempt = False
        if self._marker_exempt_active and bbox_px is not None and blob_centroid is not None:
            from ..comotion import is_co_moving
            res = is_co_moving(marker_vel, blob_vel, self._comotion_cfg)
            self.last_comotion_reason = res.reason
            if res.decidable and res.co_moving:
                comotion_exempt = True

        if self._frames_seen <= self.config.warmup_frames:
            # During warmup the zone is assumed clear: keep learning the empty
            # background so later occupancy stands out.
            cv2.accumulateWeighted(roi, self._background, 0.04)
            self.last_triggered = False
            self._consecutive_motion_frames = 0
            self._occupied_latched = False
            self._clear_streak = 0
            return []

        # Two-level occupancy with hysteresis (2026-06-22):
        #   enter  : area >= min_area_px        (decisive new occupant)
        #   stay   : area >= occupied_keep_area_px  (lower, tolerates a flat/
        #            textureless object whose diff hovers near the threshold)
        # A single dip below the enter threshold no longer releases the zone; the
        # latch is only dropped after `clear_frames_to_release` consecutive frames
        # that are below even the stay threshold. This kills the "fully-inside
        # object flickers STOP" avalanche while staying safety-biased (slow release).
        enter_area = self.config.min_area_px
        keep_area = min(self.config.occupied_keep_area_px, enter_area)
        if self._occupied_latched:
            still_occupied = area >= keep_area
        else:
            still_occupied = area >= enter_area

        if still_occupied:
            self._clear_streak = 0
            self._consecutive_motion_frames += 1
        else:
            self._clear_streak += 1
            self._consecutive_motion_frames = 0

        # Update the latch.
        if self._consecutive_motion_frames >= self.config.min_consecutive:
            self._occupied_latched = True
        if self.config.hold_static_occupant:
            if self._clear_streak >= self.config.clear_frames_to_release:
                self._occupied_latched = False
        else:
            # pure-motion opt-out: no latch, release as soon as not occupied
            self._occupied_latched = still_occupied and (
                self._consecutive_motion_frames >= self.config.min_consecutive
            )

        # SAFETY-CRITICAL: never learn the background while the zone is occupied OR
        # latched OR still inside the release-confirmation window. Only adapt when
        # we are confidently clear, so a textureless static object can never erode
        # the model frame-by-frame.
        confidently_clear = (
            not self._occupied_latched
            and not still_occupied
            and self._clear_streak >= self.config.clear_frames_to_release
        )
        if self.config.hold_static_occupant:
            learn_background = confidently_clear
        else:
            learn_background = True  # pure-motion opt-out: always adapt (static object fades like the old behaviour)
        if learn_background:
            cv2.accumulateWeighted(roi, self._background, 0.04)

        self.last_triggered = self._occupied_latched
        if not self.last_triggered:
            return []
        # Co-motion exemption: the occupant is the cart body moving with the
        # authorized marker -> not an obstacle. Safety note: this only fires when
        # exempt_active (authorized + in corridor) AND the marker is actually
        # moving AND the blob moves with it; a hand from another direction fails
        # the direction/speed test, and the independent semantic hand layer is
        # never exempted. When the cart stops, co-motion is undecidable and this
        # does not fire (caller's position-based exemption handles a settled cart).
        if comotion_exempt:
            self.last_triggered = False
            return []
        return [ObjectObservation(self.config.label, self.config.stop_pose, self.config.confidence, bbox_px=bbox_px)]

    def draw_debug(self, frame: Any) -> Any:
        if frame is None or self.last_roi_box_px is None:
            return frame
        x0, y0, x1, y1 = self.last_roi_box_px
        color = (0, 0, 255) if self.last_triggered else (0, 255, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            frame,
            f"roi_motion area={self.last_motion_area:.0f} consec={self._consecutive_motion_frames} trigger={self.last_triggered}",
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        # Draw the marker mask (cyan, dashed-look via two thin rectangles) so it's visually obvious
        # that the cart marker is being excluded from motion detection.
        if self.last_marker_mask_px is not None:
            mx0, my0, mx1, my1 = self.last_marker_mask_px
            cv2.rectangle(frame, (mx0, my0), (mx1, my1), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "marker masked",
                (mx0, max(my0 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return frame
