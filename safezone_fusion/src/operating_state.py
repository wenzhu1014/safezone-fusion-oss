from __future__ import annotations

"""System readiness gate + frame-rate monitor (safety-biased operating state).

Two real-world risks (博, 2026-06-22):

1. Startup blind spot. Right after power-on the background model is not learned
   yet and detectors are unwarmed. During this window the system MUST default to
   STOP -- it cannot yet be trusted to see a hazard ("not ready" != "clear").
   Only once warmup completes do we allow motion.

2. Frame-rate degradation. The minimum safety distance (ISO 13855) is computed
   for an assumed FPS. If the camera board CPU stalls and FPS drops, the real
   response time grows and the configured safety distance is no longer enough --
   silently. So we measure the live FPS and raise a fail-safe condition when it
   falls below the floor the distance was computed for.

Both produce a single ``not_ready`` / ``degraded`` signal the caller can OR into
its STOP decision, the same fail-safe pattern as the lens self-check. Pure
Python, no cv2; testable with an injected clock.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import time

@dataclass
class OperatingStateConfig:
    enabled: bool = True
    # Startup: require this many processed frames before leaving the STOP-default
    # readiness gate (covers background learning + detector warmup).
    warmup_frames: int = 15
    # Also require this much wall-clock since first frame (guards a burst of
    # frames arriving faster than the scene can stabilize).
    warmup_min_seconds: float = 0.5
    # Frame rate: the FPS the safety distance was computed for, and the floor
    # below which we declare degraded (fail-safe). Hysteresis avoids flapping.
    expected_fps: float = 30.0
    min_fps: float = 12.0
    recover_fps: float = 15.0
    # Window (frames) over which FPS is measured.
    fps_window: int = 30
    # Consecutive degraded/healthy samples before latching/clearing.
    degrade_after: int = 5
    recover_after: int = 10

@dataclass
class OperatingStateResult:
    ready: bool            # warmup complete
    degraded: bool         # FPS below floor (latched)
    fps: float
    frames_seen: int
    stop_required: bool    # not ready OR degraded -> caller must fail-safe STOP
    reason: str = ""

class OperatingStateMonitor:
    """Tracks readiness + live FPS; emits a fail-safe stop_required flag.

    Call :meth:`tick` once per processed frame. ``stop_required`` is True while
    warming up and whenever FPS is degraded.
    """

    def __init__(self, config: Optional[OperatingStateConfig] = None, clock=None):
        self.cfg = config or OperatingStateConfig()
        self._clock = clock or time.perf_counter
        self._t_first: Optional[float] = None
        self._stamps: Deque[float] = deque(maxlen=max(2, self.cfg.fps_window))
        self._frames = 0
        self._ready = False
        self._degraded = False
        self._degrade_streak = 0
        self._recover_streak = 0

    def reset(self) -> None:
        self._t_first = None
        self._stamps.clear()
        self._frames = 0
        self._ready = False
        self._degraded = False
        self._degrade_streak = 0
        self._recover_streak = 0

    def _measure_fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return 0.0
        return (len(self._stamps) - 1) / span

    def tick(self) -> OperatingStateResult:
        if not self.cfg.enabled:
            return OperatingStateResult(True, False, 0.0, self._frames, False, "monitor_disabled")

        now = self._clock()
        if self._t_first is None:
            self._t_first = now
        self._stamps.append(now)
        self._frames += 1

        # --- readiness gate ---
        if not self._ready:
            enough_frames = self._frames >= self.cfg.warmup_frames
            enough_time = (now - self._t_first) >= self.cfg.warmup_min_seconds
            if enough_frames and enough_time:
                self._ready = True

        # --- fps monitor (only meaningful once we have samples) ---
        fps = self._measure_fps()
        if self._ready and len(self._stamps) >= 2:
            if not self._degraded:
                if fps < self.cfg.min_fps:
                    self._degrade_streak += 1
                    self._recover_streak = 0
                    if self._degrade_streak >= self.cfg.degrade_after:
                        self._degraded = True
                else:
                    self._degrade_streak = 0
            else:
                if fps >= self.cfg.recover_fps:
                    self._recover_streak += 1
                    self._degrade_streak = 0
                    if self._recover_streak >= self.cfg.recover_after:
                        self._degraded = False
                else:
                    self._recover_streak = 0

        stop_required = (not self._ready) or self._degraded
        reason = self._build_reason(fps)
        return OperatingStateResult(
            ready=self._ready,
            degraded=self._degraded,
            fps=fps,
            frames_seen=self._frames,
            stop_required=stop_required,
            reason=reason,
        )

    def _build_reason(self, fps: float) -> str:
        if not self._ready:
            return f"warming_up:{self._frames}/{self.cfg.warmup_frames}"
        if self._degraded:
            return f"fps_degraded:{fps:.1f}<{self.cfg.min_fps}"
        return f"ok:fps={fps:.1f}"
