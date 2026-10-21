from __future__ import annotations

from typing import Any, Dict, Optional
import time

from .geometry import in_waiting_window
from .models import MarkerObservation, WaitingWindowResult

class WaitingWindowChecker:
    def __init__(self, config: Dict[str, Any]):
        self.window = config.get("waiting_window", {})
        self._anchor_marker_id: int | None = None
        self._anchor_center_px: tuple[float, float] | None = None
        self._anchor_started_at: float | None = None
        self._still_started_at: float | None = None

    def _reset_anchor(self) -> None:
        self._anchor_marker_id = None
        self._anchor_center_px = None
        self._anchor_started_at = None
        self._still_started_at = None

    def _reset_still_timer(self) -> None:
        self._still_started_at = None

    def _anchor_drift_px(self, marker: MarkerObservation) -> float:
        if marker.center_px is None or self._anchor_center_px is None:
            return 0.0
        dx = marker.center_px[0] - self._anchor_center_px[0]
        dy = marker.center_px[1] - self._anchor_center_px[1]
        return float((dx * dx + dy * dy) ** 0.5)

    def _ensure_anchor(self, marker: MarkerObservation) -> None:
        if marker.center_px is None:
            return
        if self._anchor_marker_id != marker.marker_id:
            self._anchor_marker_id = marker.marker_id
            self._anchor_center_px = marker.center_px
            self._anchor_started_at = time.time()
            return
        if self._anchor_center_px is None:
            self._anchor_center_px = marker.center_px
            self._anchor_started_at = time.time()
        if self._anchor_started_at is None:
            self._anchor_started_at = time.time()

    def _still_age_sec(self) -> float:
        if self._still_started_at is None:
            return 0.0
        return max(0.0, time.time() - self._still_started_at)

    def _anchor_age_sec(self) -> float:
        if self._anchor_started_at is None:
            return 0.0
        return max(0.0, time.time() - self._anchor_started_at)

    def _recent_anchor_radius_px(self, history_window_sec: float) -> float:
        """Max distance from anchor over recent center samples.

        The marker detector keeps timestamped center history. This checker uses
        it to answer the more stable question: has the cart stayed within a
        small anchored region for the last T seconds?
        """

        if self._anchor_center_px is None:
            return 0.0
        center_t_history = getattr(self, "_debug_no_history", None)
        # Normal path: MarkerObservation does not expose full history; the
        # OpenCV detector currently exposes recent radius via center_jitter_px.
        # Keep this method for future SensorFrame history injection, while the
        # v0.8.4 anchored_position mode combines anchor_drift + center_jitter.
        return 0.0

    def check(self, marker: Optional[MarkerObservation]) -> WaitingWindowResult:
        if marker is None or not marker.visible:
            self._reset_anchor()
            return WaitingWindowResult(False, False, "marker_not_visible")

        if not in_waiting_window(marker.pose, self.window):
            self._reset_anchor()
            return WaitingWindowResult(False, False, "cart_not_in_waiting_window")

        mode = str(self.window.get("stability_mode", "speed"))
        if mode == "relaxed":
            return WaitingWindowResult(True, True, "cart_in_window_and_stable:relaxed")

        if mode == "angled_marker":
            # Prototype camera can view the marker at a steep angle. In that case
            # OpenCV may still decode the ID correctly, but the four detected
            # corners wobble frame-to-frame, making center_speed/jitter look like
            # motion even when the cart is physically stopped. Treat continuous
            # visibility of the same authorized marker as the primary stop signal,
            # and use a larger anchor-drift limit only to reject obvious movement.
            required_frames = int(self.window.get("stable_marker_frames", self.window.get("stable_frames_min", 5)))
            stable_time_sec = float(self.window.get("stable_time_sec", 0.5))
            anchor_drift_limit = float(self.window.get("angled_anchor_drift_max_px", self.window.get("stable_anchor_drift_max_px", 48.0)))
            # Angled-marker mode intentionally tolerates corner jitter, but it
            # should not accept a cart that is slowly creeping. Require a short
            # continuous period where the marker center speed stays below a
            # conservative pixel-speed limit. This keeps the斜角抗抖动特性，
            # while making very slow movement eventually reset the stable timer.
            still_speed_limit = float(
                self.window.get(
                    "angled_still_center_speed_max_px_s",
                    self.window.get("stable_center_speed_max_px_s", 20.0),
                )
            )
            still_time_sec = float(self.window.get("angled_still_time_sec", stable_time_sec))
            sway_anchor_limit = float(self.window.get("wind_sway_anchor_drift_max_px", 0.0))
            sway_jitter_limit = float(self.window.get("wind_sway_jitter_max_px", 0.0))
            sway_min_anchor_age = float(self.window.get("wind_sway_min_anchor_age_sec", 0.3))
            self._ensure_anchor(marker)
            anchor_drift = self._anchor_drift_px(marker)
            anchor_age = self._anchor_age_sec()
            bounded_sway = False
            if anchor_drift > anchor_drift_limit:
                self._anchor_center_px = marker.center_px
                self._anchor_started_at = time.time()
                self._reset_still_timer()
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:large_drift={anchor_drift:.1f}px>{anchor_drift_limit:.1f}px",
                )
            if marker.center_speed_px_s > still_speed_limit:
                bounded_sway = (
                    sway_anchor_limit > 0.0
                    and sway_jitter_limit > 0.0
                    and anchor_age >= sway_min_anchor_age
                    and anchor_drift <= sway_anchor_limit
                    and marker.center_jitter_px <= sway_jitter_limit
                )
                if not bounded_sway:
                    self._reset_still_timer()
                    return WaitingWindowResult(
                        True,
                        False,
                        f"cart_not_stable:center_speed={marker.center_speed_px_s:.1f}px/s>{still_speed_limit:.1f}px/s",
                    )
            if self._still_started_at is None:
                self._still_started_at = time.time()
            still_age = self._still_age_sec()
            if marker.stable_frames < required_frames:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:marker_frames={marker.stable_frames}/{required_frames}",
                )
            if anchor_age < stable_time_sec:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:visible_time={anchor_age:.1f}s<{stable_time_sec:.1f}s",
                )
            if still_age < still_time_sec:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:still_time={still_age:.1f}s<{still_time_sec:.1f}s center_speed={marker.center_speed_px_s:.1f}px/s",
                )
            return WaitingWindowResult(
                True,
                True,
                f"cart_in_window_and_stable:angled_marker frames={marker.stable_frames} drift={anchor_drift:.1f}px visible={anchor_age:.1f}s still={still_age:.1f}s center_speed={marker.center_speed_px_s:.1f}px/s sway={'yes' if bounded_sway else 'no'}",
            )

        if mode == "anchored_position":
            # Stop-state judgement based on position history rather than
            # single-frame speed. This is better for outdoor/camera-distance
            # situations where instant center_v can be noisy: if the marker has
            # stayed within a small anchor region for enough time, it is stable;
            # if it leaves that region, it is moving/relocating.
            required_frames = int(self.window.get("stable_marker_frames", self.window.get("stable_frames_min", 5)))
            hold_time_sec = float(self.window.get("position_hold_time_sec", self.window.get("stable_time_sec", 0.8)))
            anchor_drift_limit = float(self.window.get("position_anchor_drift_max_px", self.window.get("wind_sway_anchor_drift_max_px", 8.0)))
            history_radius_limit = float(self.window.get("position_history_radius_max_px", self.window.get("wind_sway_jitter_max_px", 3.0)))
            center_speed_warn = float(self.window.get("position_center_speed_warn_px_s", self.window.get("angled_still_center_speed_max_px_s", 5.0)))
            self._ensure_anchor(marker)
            anchor_drift = self._anchor_drift_px(marker)
            anchor_age = self._anchor_age_sec()
            if anchor_drift > anchor_drift_limit:
                self._anchor_center_px = marker.center_px
                self._anchor_started_at = time.time()
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:position_anchor_drift={anchor_drift:.1f}px>{anchor_drift_limit:.1f}px",
                )
            if marker.center_jitter_px > history_radius_limit:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:position_radius={marker.center_jitter_px:.1f}px>{history_radius_limit:.1f}px",
                )
            if marker.stable_frames < required_frames:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:marker_frames={marker.stable_frames}/{required_frames}",
                )
            if anchor_age < hold_time_sec:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:position_hold_time={anchor_age:.1f}s<{hold_time_sec:.1f}s",
                )
            speed_note = "fast_local_sway" if marker.center_speed_px_s > center_speed_warn else "calm"
            return WaitingWindowResult(
                True,
                True,
                f"cart_in_window_and_stable:anchored_position frames={marker.stable_frames} drift={anchor_drift:.1f}px radius={marker.center_jitter_px:.1f}px hold={anchor_age:.1f}s center_speed={marker.center_speed_px_s:.1f}px/s note={speed_note}",
            )

        if mode == "jitter":
            jitter_limit = float(self.window.get("stable_center_jitter_max_px", 8.0))
            center_speed_limit = float(self.window.get("stable_center_speed_max_px_s", 20.0))
            anchor_drift_limit = float(self.window.get("stable_anchor_drift_max_px", 12.0))
            stable = marker.center_jitter_px <= jitter_limit and marker.center_speed_px_s <= center_speed_limit
            if not stable:
                if marker.center_speed_px_s > center_speed_limit:
                    return WaitingWindowResult(
                        True,
                        False,
                        f"cart_not_stable:center_speed={marker.center_speed_px_s:.1f}px/s>{center_speed_limit:.1f}px/s",
                    )
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:jitter={marker.center_jitter_px:.1f}px>{jitter_limit:.1f}px",
                )
            self._ensure_anchor(marker)
            anchor_drift = self._anchor_drift_px(marker)
            if anchor_drift > anchor_drift_limit:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:anchor_drift={anchor_drift:.1f}px>{anchor_drift_limit:.1f}px",
                )
            return WaitingWindowResult(
                True,
                True,
                f"cart_in_window_and_stable:jitter={marker.center_jitter_px:.1f}px center_speed={marker.center_speed_px_s:.1f}px/s drift={anchor_drift:.1f}px",
            )

        if mode == "smoothed_speed":
            speed_limit = float(self.window.get("stable_speed_max_mps", 0.01))
            stable = marker.smoothed_speed_mps <= speed_limit
            if not stable:
                return WaitingWindowResult(
                    True,
                    False,
                    f"cart_not_stable:smooth_speed={marker.smoothed_speed_mps:.3f}>{speed_limit:.3f}",
                )
            return WaitingWindowResult(True, True, f"cart_in_window_and_stable:smooth_speed={marker.smoothed_speed_mps:.3f}")

        speed_limit = float(self.window.get("stable_speed_max_mps", 0.01))
        stable = marker.speed_mps <= speed_limit
        if not stable:
            return WaitingWindowResult(True, False, f"cart_not_stable:speed={marker.speed_mps:.3f}")

        return WaitingWindowResult(True, True, "cart_in_window_and_stable")
