"""Safety watchdog: fail-safe on a stale/lost signal link.

WHY THIS LIVES ON THE CONSUMER SIDE
-----------------------------------
If the camera board crashes, loses power, or its cable is pulled, it cannot
"announce" that it died -- a dead board sends nothing. So the only safe design
is for the CONSUMER (Android main board / PLC) to watch the incoming signal
stream and stop the mechanism when signals go stale:

    last fresh signal older than `timeout` => FAIL-SAFE STOP

This module is the reference watchdog logic. The Android/PLC side should
implement the same rule so behaviour matches. It is pure and deterministic
(time is injected), so it is fully testable without hardware.

PAIRED PRODUCER REQUIREMENT
---------------------------
For the watchdog to work, the producer (SafeZone) must emit at a minimum
heartbeat rate even when the decision does not change (e.g. long IDLE). A
consumer that sees no new ``seq`` cannot tell "all calm" from "board dead", so
silence MUST be treated as danger. SafetyHeartbeat (below) helps the producer
guarantee that minimum rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class WatchdogState:
    """Result of one watchdog evaluation."""

    fail_safe: bool          # True => consumer must force a STOP now
    reason: str              # human-readable cause
    age_sec: Optional[float] # seconds since last fresh signal (None if never seen)
    last_seq: Optional[int]  # last accepted sequence number

class SafetyWatchdog:
    """Watches signal freshness and sequence continuity.

    Fail-safe triggers when EITHER:
      - no signal has ever been seen (startup grace excepted), OR
      - the newest signal is older than ``timeout_sec`` (link stale / board dead).

    Sequence regression/duplication is reported (possible replay / restart) but
    does NOT by itself trigger fail-safe -- staleness is the safety condition.
    """

    def __init__(self, timeout_sec: float = 0.5, startup_grace_sec: float = 2.0):
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        self.timeout_sec = float(timeout_sec)
        self.startup_grace_sec = float(startup_grace_sec)
        self._last_seq: Optional[int] = None
        self._last_rx_monotonic: Optional[float] = None
        self._first_check_monotonic: Optional[float] = None

    def record(self, seq: int, now_monotonic: float) -> None:
        """Register receipt of a signal with sequence ``seq`` at ``now_monotonic``.

        Out-of-order/duplicate seq are ignored for freshness (we only advance on
        a strictly newer seq), but receiving ANY valid newer frame refreshes the
        liveness timer.
        """
        if self._last_seq is None or seq > self._last_seq:
            self._last_seq = seq
            self._last_rx_monotonic = now_monotonic

    def check(self, now_monotonic: float) -> WatchdogState:
        """Evaluate fail-safe condition at time ``now_monotonic`` (monotonic clock)."""
        if self._first_check_monotonic is None:
            self._first_check_monotonic = now_monotonic

        if self._last_rx_monotonic is None:
            # Never received a signal. Allow a short startup grace, then fail safe.
            since_start = now_monotonic - self._first_check_monotonic
            if since_start <= self.startup_grace_sec:
                return WatchdogState(False, "startup_grace_no_signal_yet", None, None)
            return WatchdogState(True, "no_signal_received", None, None)

        age = now_monotonic - self._last_rx_monotonic
        if age > self.timeout_sec:
            return WatchdogState(True, "signal_stale", age, self._last_seq)
        return WatchdogState(False, "ok", age, self._last_seq)

class SafetyHeartbeat:
    """Producer-side helper: should we emit now to keep the link alive?

    Even if the decision is unchanged, the producer must emit at least every
    ``min_interval_sec`` so the consumer watchdog never mistakes calm for death.
    Call should_emit(now) each loop; it returns True when either the decision
    changed or the heartbeat interval elapsed.
    """

    def __init__(self, min_interval_sec: float = 0.2):
        if min_interval_sec <= 0:
            raise ValueError("min_interval_sec must be > 0")
        self.min_interval_sec = float(min_interval_sec)
        self._last_emit_monotonic: Optional[float] = None
        self._last_key: Optional[tuple] = None

    def should_emit(self, now_monotonic: float, decision_key: tuple) -> bool:
        if self._last_emit_monotonic is None:
            self._last_emit_monotonic = now_monotonic
            self._last_key = decision_key
            return True
        changed = decision_key != self._last_key
        elapsed = now_monotonic - self._last_emit_monotonic
        if changed or elapsed >= self.min_interval_sec:
            self._last_emit_monotonic = now_monotonic
            self._last_key = decision_key
            return True
        return False
