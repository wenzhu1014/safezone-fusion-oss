"""Pluggable safety-signal output layer for SafeZone.

SafeZone decides STOP / ALLOW / WARNING / IDLE on the camera board. That signal
must reach the cabinet's Android main board and ultimately the PLC, which stops
the servo pulse train to halt the mechanism. The *transport* (MQTT / serial /
TCP / file) is not finalised yet, so this layer keeps the algorithm decoupled
from it:

    ActuatorOutput  ->  SafetySignalEmitter  ->  [one or more SafetySignalSink]

Swap or add a transport by implementing SafetySignalSink. The core logic and
the emitter never change. A FileSink works today (no external deps), so the
output contract is testable and observable before the Android/PLC side exists.

Signal schema (JSON), designed around what a PLC actually needs:

    {
      "schema": "safezone.safety_signal.v1",
      "seq": 1234,                # monotonic sequence; detect dropped/stale msgs
      "ts": 1750000000.123,       # unix seconds
      "stop": true,               # THE field: must the mechanism stop NOW?
      "motor": "STOP",            # MotorAction enum value
      "decision": "STOP",         # raw fusion decision
      "reason": "human_in_stop_zone",
      "voice": "检测到人员或肢体，请立即停止。",
      "source": "safezone-camera"
    }

PLC-facing contract: act on ``stop``. ``stop=true`` => stop sending pulses /
drop enable. Everything else is context/telemetry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from .interfaces import ActuatorOutput, MotorAction

SCHEMA = "safezone.safety_signal.v1"

# Motor actions that mean "the mechanism must NOT be moving freely / must hold
# or stop". Used to derive the single boolean the PLC cares about.
_STOP_MOTOR_ACTIONS = {MotorAction.STOP, MotorAction.LOCK_STILL}

def build_signal(output: ActuatorOutput, seq: int, *, source: str = "safezone-camera") -> dict[str, Any]:
    """Translate an ActuatorOutput into the wire signal dict."""
    motor = output.motor
    motor_value = motor.value if hasattr(motor, "value") else str(motor)
    stop = motor in _STOP_MOTOR_ACTIONS or (output.raw_decision == "STOP")
    return {
        "schema": SCHEMA,
        "seq": seq,
        "ts": round(time.time(), 3),
        "stop": bool(stop),
        "motor": motor_value,
        "decision": output.raw_decision,
        "reason": output.reason,
        "voice": output.voice,
        "source": source,
    }

class SafetySignalSink(Protocol):
    """A transport that delivers one safety signal."""

    def emit(self, signal: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...

class StdoutSink:
    """Prints signals as compact JSON. Useful for bring-up / debugging."""

    def emit(self, signal: dict[str, Any]) -> None:
        print("[safety-signal] " + json.dumps(signal, ensure_ascii=False))

    def close(self) -> None:
        pass

class JsonlFileSink:
    """Appends each signal as one JSON line. Works today, zero deps.

    The Android/PLC bring-up can tail this file, or we keep it as an audit log
    even after a live transport is added.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, signal: dict[str, Any]) -> None:
        self._fh.write(json.dumps(signal, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

class LatestJsonSink:
    """Atomically overwrites a single JSON file with the latest signal.

    Lets any consumer (Android poll, another process) read the current safety
    state without parsing a growing log. Write-rename keeps reads atomic.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, signal: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(signal, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def close(self) -> None:
        pass

class MqttSink:
    """Publishes signals to an MQTT topic. Lazy-imports paho-mqtt.

    Deliberately optional: if paho-mqtt is not installed, constructing this
    sink raises a clear error, so selecting MQTT is explicit intent. Transport
    config (host/port/topic) is passed in, never hard-coded.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1883,
                 topic: str = "safezone/safety_signal", qos: int = 1,
                 client_id: str = "safezone-camera"):
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "MqttSink needs paho-mqtt: pip install paho-mqtt. "
                f"Import failed: {exc}"
            ) from exc
        self._topic = topic
        self._qos = qos
        self._client = mqtt.Client(client_id=client_id)
        self._client.connect(host, port, keepalive=30)
        self._client.loop_start()

    def emit(self, signal: dict[str, Any]) -> None:
        self._client.publish(self._topic, json.dumps(signal, ensure_ascii=False), qos=self._qos)

    def close(self) -> None:  # pragma: no cover - env dependent
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

class SafetySignalEmitter:
    """Fans one ActuatorOutput out to every configured sink as a versioned signal.

    - Monotonic ``seq`` lets the PLC side detect dropped or stale messages
      (important for a safety channel).
    - A failing sink never breaks the others or the control loop: emit() is
      best-effort per sink and swallows sink exceptions (logged via on_error).
    """

    def __init__(self, sinks: Iterable[SafetySignalSink], *, source: str = "safezone-camera",
                 on_error=None):
        self._sinks = list(sinks)
        self._seq = 0
        self._source = source
        self._on_error = on_error

    def emit(self, output: ActuatorOutput) -> dict[str, Any]:
        self._seq += 1
        signal = build_signal(output, self._seq, source=self._source)
        for sink in self._sinks:
            try:
                sink.emit(signal)
            except Exception as exc:  # best-effort: one bad sink must not stop others
                if self._on_error is not None:
                    self._on_error(sink, exc)
        return signal

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass

def build_sinks_from_spec(spec: Optional[str], *, audit_path: str | Path | None = None) -> list[SafetySignalSink]:
    """Build sinks from a simple CLI-friendly spec string.

    spec examples:
      None / ""              -> []  (no live signal output)
      "stdout"               -> [StdoutSink]
      "file:out/sig.jsonl"   -> [JsonlFileSink]
      "latest:out/now.json"  -> [LatestJsonSink]
      "mqtt:127.0.0.1:1883:safezone/safety_signal" -> [MqttSink]
      "stdout,latest:out/now.json" -> multiple, comma-separated

    If audit_path is given, a JsonlFileSink is always appended so there is a
    durable record regardless of the live transport chosen.
    """
    sinks: list[SafetySignalSink] = []
    for raw in (spec or "").split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "stdout":
            sinks.append(StdoutSink())
        elif token.startswith("file:"):
            sinks.append(JsonlFileSink(token[len("file:"):]))
        elif token.startswith("latest:"):
            sinks.append(LatestJsonSink(token[len("latest:"):]))
        elif token.startswith("mqtt:"):
            parts = token.split(":")
            host = parts[1] if len(parts) > 1 and parts[1] else "127.0.0.1"
            port = int(parts[2]) if len(parts) > 2 and parts[2] else 1883
            topic = parts[3] if len(parts) > 3 and parts[3] else "safezone/safety_signal"
            sinks.append(MqttSink(host=host, port=port, topic=topic))
        else:
            raise ValueError(f"unknown safety-signal sink spec: {token!r}")
    if audit_path is not None:
        sinks.append(JsonlFileSink(audit_path))
    return sinks
