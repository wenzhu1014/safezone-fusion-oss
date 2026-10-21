from __future__ import annotations

"""Lens blinding / occlusion self-check (IEC 61496 style integrity monitoring).

Safety rationale
----------------
A camera that is covered, smeared, blinded by glare, or out of focus produces a
frame in which "nothing is visible" -- and a naive detector reads that as
"nothing is there", i.e. *safe to dock*. That is the classic "invisible = safe"
blind spot and it is the inverse of a safety device: the most dangerous failure
is the one that looks calm.

This module treats the *image itself* as something that must pass a health
check every frame, independent of any object/marker detection. If the frame is:

* all black (lens covered / no signal),
* blown out white (glare / IR flood saturation / pointed at a light),
* flat / featureless (defocus, smeared lens, sensor stuck on one value),

then the camera cannot be trusted to see a hazard, so we force a fail-safe STOP.

The monitor is intentionally cheap (a few numpy reductions) and has **no
dependency on the fusion engine**, so it can also run on the consumer/producer
side close to the sensor, the same way the safety watchdog does.

Hysteresis
----------
A single bad frame (one dropped/garbled grab, a momentary reflection) must not
flip the system, and once we have declared the lens blinded we must not release
on a single good frame either. So we require ``enter_frames`` consecutive bad
frames to latch BLINDED, and ``release_frames`` consecutive healthy frames to
clear it -- biased so that releasing is slower than latching (safety prefers to
stay stopped).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

class LensFault(str, Enum):
    NONE = "none"
    COVERED_DARK = "covered_dark"          # frame essentially black: lens covered / no signal
    GLARE_SATURATED = "glare_saturated"    # frame blown out white: glare / IR flood / pointed at light
    FLAT_FEATURELESS = "flat_featureless"  # defocus / smear / stuck sensor: no spatial detail

@dataclass
class LensHealthResult:
    blinded: bool                 # latched decision after hysteresis -- this is what drives fail-safe STOP
    fault: LensFault              # latched fault category
    instant_bad: bool             # this single frame looked bad (pre-hysteresis)
    instant_fault: LensFault      # this single frame's fault category (pre-hysteresis)
    metrics: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

@dataclass
class LensHealthConfig:
    enabled: bool = True
    # Downscale the long edge to at most this many px before computing stats.
    # Lens faults (dark/glare/defocus) are global/low-frequency, so a small
    # frame is plenty and keeps this safety check cheap (it sits on the
    # response-time critical path). 0 disables downscaling.
    downscale_long_edge: int = 320
    # A frame is "dark" when its mean intensity (0-255) is at or below this.
    dark_mean_max: float = 12.0
    # A frame is "saturated" when its mean intensity is at or above this ...
    bright_mean_min: float = 243.0
    # ... AND most pixels are near-max (fraction of pixels >= bright_pixel_value).
    bright_pixel_value: float = 250.0
    bright_pixel_fraction_min: float = 0.90
    # A frame is "flat/featureless" when both global stddev and local detail are
    # below threshold. stddev guards against a uniform grey field; the Laplacian
    # (or gradient) variance guards against a defocused / smeared image.
    flat_stddev_max: float = 4.0
    flat_detail_max: float = 6.0
    # Hysteresis: latch BLINDED after this many consecutive bad frames; clear
    # only after this many consecutive healthy frames (release slower = safer).
    enter_frames: int = 3
    release_frames: int = 8

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "LensHealthConfig":
        section = (config or {}).get("lens_health", {}) if config else {}
        base = cls()
        if not section:
            return base
        return cls(
            enabled=bool(section.get("enabled", base.enabled)),
            downscale_long_edge=int(section.get("downscale_long_edge", base.downscale_long_edge)),
            dark_mean_max=float(section.get("dark_mean_max", base.dark_mean_max)),
            bright_mean_min=float(section.get("bright_mean_min", base.bright_mean_min)),
            bright_pixel_value=float(section.get("bright_pixel_value", base.bright_pixel_value)),
            bright_pixel_fraction_min=float(
                section.get("bright_pixel_fraction_min", base.bright_pixel_fraction_min)
            ),
            flat_stddev_max=float(section.get("flat_stddev_max", base.flat_stddev_max)),
            flat_detail_max=float(section.get("flat_detail_max", base.flat_detail_max)),
            enter_frames=int(section.get("enter_frames", base.enter_frames)),
            release_frames=int(section.get("release_frames", base.release_frames)),
        )

def _to_gray(frame, downscale_long_edge: int = 0):
    """Best-effort convert any frame (BGR / mono / None) to a 2D float array.

    Returns None when the frame is unusable (None or empty). We avoid a hard
    OpenCV dependency: a 3-channel array is reduced by simple mean over the last
    axis, which is good enough for intensity statistics. When
    ``downscale_long_edge`` > 0 the image is cheaply subsampled (strided) so the
    long edge is at most that many pixels -- this keeps the check off the
    response-time critical path without changing the fault decisions.
    """
    if frame is None:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    arr = np.asarray(frame)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    elif arr.ndim != 2:
        return None
    if downscale_long_edge and downscale_long_edge > 0:
        h, w = arr.shape[:2]
        long_edge = max(h, w)
        if long_edge > downscale_long_edge:
            step = int(long_edge // downscale_long_edge) + 1
            if step > 1:
                arr = arr[::step, ::step]
    return arr.astype("float32")

def _detail_metric(gray) -> float:
    """A cheap focus/detail measure: variance of a 4-neighbour Laplacian.

    High on a sharp, textured scene; near zero on a defocused / smeared / flat
    frame. Implemented with numpy slicing so we do not require cv2 here.
    """
    try:
        import numpy as np
    except Exception:
        return 0.0
    if gray is None or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())

def analyze_frame(frame, config: LensHealthConfig) -> tuple[bool, LensFault, Dict[str, float]]:
    """Stateless single-frame health check.

    Returns (is_bad, fault, metrics). A missing/empty frame counts as bad
    (COVERED_DARK) -- a dead capture must not be read as a clear scene.
    """
    gray = _to_gray(frame, getattr(config, "downscale_long_edge", 0))
    if gray is None:
        return True, LensFault.COVERED_DARK, {"mean": 0.0, "stddev": 0.0, "detail": 0.0, "bright_frac": 0.0}

    try:
        import numpy as np
    except Exception:
        # Without numpy we cannot verify the frame; fail safe.
        return True, LensFault.FLAT_FEATURELESS, {}

    mean = float(gray.mean())
    stddev = float(gray.std())
    bright_frac = float((gray >= config.bright_pixel_value).mean())
    detail = _detail_metric(gray)
    metrics = {
        "mean": mean,
        "stddev": stddev,
        "detail": detail,
        "bright_frac": bright_frac,
    }

    # Order matters: dark and saturated are the strongest signals.
    if mean <= config.dark_mean_max:
        return True, LensFault.COVERED_DARK, metrics
    if mean >= config.bright_mean_min and bright_frac >= config.bright_pixel_fraction_min:
        return True, LensFault.GLARE_SATURATED, metrics
    if stddev <= config.flat_stddev_max and detail <= config.flat_detail_max:
        return True, LensFault.FLAT_FEATURELESS, metrics
    return False, LensFault.NONE, metrics

class LensHealthMonitor:
    """Stateful lens integrity monitor with latch + hysteresis.

    Call :meth:`update` once per captured frame. ``result.blinded`` is the
    latched, debounced decision that should drive a fail-safe STOP; treat
    ``blinded=True`` exactly like an obstacle in the stop zone.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if isinstance(config, LensHealthConfig):
            self.cfg = config
        else:
            self.cfg = LensHealthConfig.from_config(config)
        self._blinded = False
        self._fault = LensFault.NONE
        self._bad_streak = 0
        self._good_streak = 0

    @property
    def blinded(self) -> bool:
        return self._blinded

    def reset(self) -> None:
        self._blinded = False
        self._fault = LensFault.NONE
        self._bad_streak = 0
        self._good_streak = 0

    def update(self, frame) -> LensHealthResult:
        if not self.cfg.enabled:
            return LensHealthResult(
                blinded=False,
                fault=LensFault.NONE,
                instant_bad=False,
                instant_fault=LensFault.NONE,
                reason="lens_health_disabled",
            )

        is_bad, fault, metrics = analyze_frame(frame, self.cfg)

        if is_bad:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            self._good_streak += 1
            self._bad_streak = 0

        if not self._blinded:
            if self._bad_streak >= self.cfg.enter_frames:
                self._blinded = True
                self._fault = fault
        else:
            # Keep reporting the latched fault; refresh it if still bad.
            if is_bad:
                self._fault = fault
            if self._good_streak >= self.cfg.release_frames:
                self._blinded = False
                self._fault = LensFault.NONE

        reason = self._build_reason(is_bad, fault)
        return LensHealthResult(
            blinded=self._blinded,
            fault=self._fault if self._blinded else LensFault.NONE,
            instant_bad=is_bad,
            instant_fault=fault,
            metrics=metrics,
            reason=reason,
        )

    def _build_reason(self, is_bad: bool, fault: LensFault) -> str:
        if self._blinded:
            return f"lens_blinded:{self._fault.value}"
        if is_bad:
            return f"lens_degrading:{fault.value}:{self._bad_streak}/{self.cfg.enter_frames}"
        return "lens_ok"

def apply_lens_failsafe(actuator, lens: LensHealthResult):
    """Force a fail-safe STOP on the actuator output when the lens is blinded.

    This runs *outside* and *after* the fusion engine on purpose: a blinded
    camera cannot be trusted to have produced a meaningful decision, so we
    override whatever it decided. The override is duck-typed against
    ``ActuatorOutput`` (sets ``motor`` to a STOP-like action and stamps the
    reason), so this module stays free of an import cycle with ``interfaces``.

    Returns the (possibly mutated) actuator for convenient chaining. When the
    lens is healthy the actuator is returned untouched.
    """
    if lens is None or not lens.blinded:
        return actuator
    # Resolve a STOP value without importing MotorAction (avoid cycle): prefer
    # the existing motor enum's STOP member if present, else fall back to a
    # plain string "STOP".
    stop_value = "STOP"
    motor = getattr(actuator, "motor", None)
    stop_member = getattr(type(motor), "STOP", None) if motor is not None else None
    if stop_member is not None:
        stop_value = stop_member
    try:
        actuator.motor = stop_value
        actuator.reason = lens.reason or "lens_blinded"
        # Mark in debug so downstream/HUD can show it distinctly.
        dbg = getattr(actuator, "debug", None)
        if isinstance(dbg, dict):
            dbg["lens_failsafe"] = True
            dbg["lens_fault"] = lens.fault.value
    except Exception:
        pass
    return actuator
