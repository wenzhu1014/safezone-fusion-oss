from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

import cv2
import numpy as np

from ..models import MarkerObservation, Pose
from .interfaces import CameraPacket, MarkerDetector

@dataclass
class CameraCalibration:
    camera_matrix: Any | None = None
    dist_coeffs: Any | None = None
    # Transform from camera coordinates to cabinet coordinates.
    # v0.6 defaults to identity-like mapping for prototype.
    camera_to_cabinet: Any | None = None

class OpenCVArucoMarkerDetector(MarkerDetector):
    """OpenCV ArUco detector prototype.

    v0.8 supports:
    - marker ID detection from single camera (`rgb`/`mono`/`nir`) or stereo (`left_rgb`/`right_rgb`)
    - optional pose estimation if camera intrinsics and marker_size_m are provided
    - stable frame counting and rough speed estimation

    Coordinate note:
    - Without calibration, pose is a placeholder based on image center.
    - With calibration, tvec is mapped to a simplified cabinet pose:
      x=tvec[0], y=-tvec[1], z=tvec[2]. Real deployment should use full extrinsic calibration.
    """

    def __init__(
        self,
        dictionary_name: str = "DICT_6X6_250",
        marker_size_m: float = 0.10,
        calibration: CameraCalibration | None = None,
        target_marker_ids: set[int] | None = None,
        speed_window: int = 7,
        jitter_window: int = 7,
    ):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module is unavailable. Install opencv-contrib-python.")
        self.dictionary_name = dictionary_name
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_size_m = marker_size_m
        self.calibration = calibration or CameraCalibration()
        self.target_marker_ids = target_marker_ids
        self.speed_window = max(1, int(speed_window))
        self.jitter_window = max(1, int(jitter_window))
        self._last_id: Optional[int] = None
        self._stable_frames = 0
        self._last_pose: Optional[Pose] = None
        self._last_timestamp: Optional[float] = None
        # Marker coasting (2026-06-22): a single dropped frame must NOT reset the
        # stability counter, or the whole cart-verification ladder restarts and
        # the UI flickers IDLE/SCANNING/ALLOW even with a perfectly still marker.
        # Tolerate a few consecutive misses by re-emitting the last good marker
        # (flagged coasting) before declaring the marker truly gone.
        self._last_marker: Optional[MarkerObservation] = None
        self._coast_misses: int = 0
        self._MAX_COAST_FRAMES: int = 4
        self._speed_history: deque[float] = deque(maxlen=self.speed_window)
        self._center_history: deque[tuple[float, float]] = deque(maxlen=self.jitter_window)
        self._center_t_history: deque[tuple[float, float, float]] = deque(maxlen=self.jitter_window)
        self.last_detected_ids: list[int] = []
        self.last_rejected_count: int = 0
        self.last_used_mirror: bool = False
        self.last_corners: Any | None = None
        self.last_ids_array: Any | None = None
        self.last_stream_name: str | None = None
        # Fallback hysteresis (2026-06-22): prefer RGB whenever it detects, and
        # only switch to a mono/NIR fallback after RGB has failed for this many
        # consecutive frames. Prevents rgb<->mono flip-flop (parallax ~88px)
        # from injecting fake jitter/speed into stop-stability judging.
        self._rgb_fail_streak: int = 0
        self._FALLBACK_HOLD_FRAMES: int = 6

    def detect(self, packet: CameraPacket) -> MarkerObservation | None:
        # Try streams in priority order. Important: when RGB exists but is blocked/
        # overexposed and detection fails, continue to mono/NIR rather than stopping
        # at the first non-null image. This makes dual-camera fallback actually work.
        candidates = self._candidate_images(packet)
        if not candidates:
            return None

        image = None
        corners = ids = rejected = None
        used_mirror = False
        used_stream = None
        last_rejected_count = 0

        # Detect across every available stream this frame, remembering each
        # stream's result, so the hysteresis below can pick which one to use
        # rather than blindly taking the first that happens to succeed.
        RGB_CLASS = ("rgb", "left_rgb", "right_rgb")
        detections: dict[str, tuple] = {}
        for stream_name, stream_image in candidates:
            img = cv2.imread(stream_image) if isinstance(stream_image, str) else stream_image
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            c, i, r = self.detector.detectMarkers(gray)
            rejected_count = 0 if r is None else len(r)
            mirror = False

            # Many webcam preview paths are mirrored. ArUco IDs are not mirror
            # invariant, so try a horizontal flip before declaring failure.
            if i is None or len(i) == 0:
                mirrored = cv2.flip(gray, 1)
                mc, mi, mr = self.detector.detectMarkers(mirrored)
                if mi is not None and len(mi) > 0:
                    c = self._mirror_corners_to_original(mc, gray.shape[1])
                    i, r = mi, mr
                    mirror = True
                    rejected_count = 0 if r is None else len(r)

            if i is not None and len(i) > 0:
                detections[stream_name] = (img, c, i, r, mirror, rejected_count)

        # Hysteresis stream selection.
        rgb_hit = next((s for s in RGB_CLASS if s in detections), None)
        if rgb_hit is not None:
            # RGB sees the marker -> always use it (single coordinate basis).
            self._rgb_fail_streak = 0
            chosen = rgb_hit
        else:
            # RGB failed this frame. Hold briefly before trusting a fallback
            # stream, so a single dropped RGB frame does not jump coordinates.
            self._rgb_fail_streak += 1
            fallback = next((s for s, _ in candidates if s in detections), None)
            if fallback is not None and self._rgb_fail_streak >= self._FALLBACK_HOLD_FRAMES:
                chosen = fallback
            else:
                chosen = None  # treat as momentary no-detection (no coord jump)

        if chosen is not None:
            img, c, i, r, mirror, rejected_count = detections[chosen]
            image = img
            corners, ids, rejected = c, i, r
            used_mirror = mirror
            used_stream = chosen
            last_rejected_count = rejected_count

        self.last_used_mirror = used_mirror
        self.last_rejected_count = last_rejected_count
        # Stream-change coordinate isolation (2026-06-22): the two cameras see the
        # marker ~88px apart (parallax). If we switch streams we must NOT subtract
        # the previous stream's center history from the new stream's coordinates,
        # or jitter/speed spike with a fake 800+px/s reading. Reset the position
        # histories on any stream change so each stream is measured against itself.
        if used_stream is not None and used_stream != self.last_stream_name:
            self._center_history.clear()
            self._center_t_history.clear()
            self._speed_history.clear()
        self.last_stream_name = used_stream

        if ids is None or len(ids) == 0 or image is None:
            # Coasting: tolerate a few dropped frames before declaring the marker
            # gone, so transient detection misses don't reset stability/UI.
            if self._last_marker is not None and self._coast_misses < self._MAX_COAST_FRAMES:
                self._coast_misses += 1
                self.last_detected_ids = [self._last_marker.marker_id]
                # keep last_corners/ids_array so debug overlay still draws steadily
                lm = self._last_marker
                return MarkerObservation(
                    marker_id=lm.marker_id,
                    pose=lm.pose,
                    stable_frames=self._stable_frames,
                    speed_mps=lm.speed_mps,
                    visible=False,            # honest: this frame did not see it
                    smoothed_speed_mps=lm.smoothed_speed_mps,
                    center_px=lm.center_px,
                    center_jitter_px=lm.center_jitter_px,
                    center_speed_px_s=lm.center_speed_px_s,
                    calibrated_pose=lm.calibrated_pose,
                    coasting=True,
                )
            # Truly gone (exceeded coast budget): hard reset.
            self._last_id = None
            self._stable_frames = 0
            self._last_marker = None
            self._speed_history.clear()
            self._center_history.clear()
            self._center_t_history.clear()
            self.last_detected_ids = []
            self.last_corners = None
            self.last_ids_array = None
            return None
        self._coast_misses = 0

        marker_ids = [int(x[0]) for x in ids]
        self.last_detected_ids = marker_ids
        self.last_corners = corners
        self.last_ids_array = ids
        selected_index = 0
        if self.target_marker_ids:
            selected_index = -1
            for i, mid in enumerate(marker_ids):
                if mid in self.target_marker_ids:
                    selected_index = i
                    break
            if selected_index < 0:
                return None

        marker_id = marker_ids[selected_index]
        marker_corners = corners[selected_index]
        pose, calibrated_pose = self._estimate_pose(marker_corners, image)
        center_px = self._marker_center(marker_corners)

        if marker_id == self._last_id:
            self._stable_frames += 1
        else:
            self._last_id = marker_id
            self._stable_frames = 1
            self._speed_history.clear()
            self._center_history.clear()
            self._center_t_history.clear()

        speed = self._estimate_speed(pose, packet.timestamp)
        self._speed_history.append(speed)
        self._center_history.append(center_px)
        self._center_t_history.append((center_px[0], center_px[1], packet.timestamp))
        smoothed_speed = float(median(self._speed_history)) if self._speed_history else speed
        center_jitter = self._center_jitter()
        center_speed = self._center_speed_px_s()
        self._last_pose = pose
        self._last_timestamp = packet.timestamp

        observation = MarkerObservation(
            marker_id=marker_id,
            pose=pose,
            stable_frames=self._stable_frames,
            speed_mps=speed,
            visible=True,
            smoothed_speed_mps=smoothed_speed,
            center_px=center_px,
            center_jitter_px=center_jitter,
            center_speed_px_s=center_speed,
            calibrated_pose=calibrated_pose,
        )
        self._last_marker = observation
        return observation

    def _candidate_images(self, packet: CameraPacket) -> list[tuple[str, Any]]:
        """Return marker-detection streams in fixed priority order.

        ArUco works on grayscale, so mono/NIR is a valid fallback. detect() tries
        every candidate until one detects a marker. In dual-camera mode RGB may
        be physically present but blocked, while NIR still sees the printed
        marker -- so we must be able to fall through to it.

        Order is FIXED (rgb first). The flip-flop/jitter problem is solved in
        detect() via fallback hysteresis (see _FALLBACK_HOLD_FRAMES), not by
        reordering here: two cameras see the same marker ~88px apart (parallax),
        so we must not switch to a fallback stream just because it also happens
        to detect the marker on a frame where RGB momentarily failed.
        """

        candidates = [
            ("rgb", packet.rgb),
            ("left_rgb", packet.left_rgb),
            ("right_rgb", packet.right_rgb),
            ("mono", packet.mono),
            ("nir", packet.nir),
        ]
        return [(name, image) for name, image in candidates if image is not None]

    def _select_image(self, packet: CameraPacket) -> Any | None:
        """Backward-compatible helper for tests / external callers.

        Returns the first available image only. Marker detection itself uses
        _candidate_images() and falls through on detection failure.
        """
        candidates = self._candidate_images(packet)
        return candidates[0][1] if candidates else None

    def draw_debug(self, image: Any) -> Any:
        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return image

    def _estimate_pose(self, marker_corners: Any, image: Any) -> tuple[Pose, bool]:
        cm = self.calibration.camera_matrix
        dc = self.calibration.dist_coeffs
        if cm is not None and dc is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], self.marker_size_m, np.asarray(cm, dtype=np.float64), np.asarray(dc, dtype=np.float64)
            )
            t = tvecs[0][0]
            # Simplified camera->cabinet placeholder. Replace with extrinsic transform in real deployment.
            return Pose(x=float(t[0]), y=float(-t[1]), z=float(t[2]), yaw_deg=0.0), True

        # No calibration fallback: convert marker center in image to normalized pseudo pose.
        pts = marker_corners.reshape(-1, 2)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        h, w = image.shape[:2]
        x = (cx - w / 2) / max(w, 1)
        y = 1.0 - cy / max(h, 1)
        z = 0.5
        return Pose(x=float(x), y=float(y), z=float(z), yaw_deg=0.0), False

    def _marker_center(self, marker_corners: Any) -> tuple[float, float]:
        pts = marker_corners.reshape(-1, 2)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def _mirror_corners_to_original(self, corners: Any, image_width: int) -> list[Any]:
        mapped = []
        for c in corners:
            cc = np.array(c, copy=True)
            cc[..., 0] = (image_width - 1) - cc[..., 0]
            mapped.append(cc)
        return mapped

    def _center_jitter(self) -> float:
        if len(self._center_history) <= 1:
            return 0.0
        xs = [p[0] for p in self._center_history]
        ys = [p[1] for p in self._center_history]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return float(max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in self._center_history))

    def _center_speed_px_s(self) -> float:
        if len(self._center_t_history) <= 1:
            return 0.0
        x0, y0, t0 = self._center_t_history[0]
        x1, y1, t1 = self._center_t_history[-1]
        dt = max(t1 - t0, 1e-6)
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt)

    def _estimate_speed(self, pose: Pose, timestamp: float) -> float:
        if self._last_pose is None or self._last_timestamp is None:
            return 0.0
        dt = max(timestamp - self._last_timestamp, 1e-6)
        dx = pose.x - self._last_pose.x
        dy = pose.y - self._last_pose.y
        dz = pose.z - self._last_pose.z
        return float((dx * dx + dy * dy + dz * dz) ** 0.5 / dt)
