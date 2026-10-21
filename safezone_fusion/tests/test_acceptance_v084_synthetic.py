from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import safezone_fusion.src.waiting_window as waiting_window_module
from safezone_fusion.src.fusion import FusionDecisionEngine
from safezone_fusion.src.models import Decision, MarkerObservation, ObjectObservation, Pose, SensorFrame

def base_config(environment: str = "indoor") -> dict:
    cfg = {
        "authorized_carts": {
            "CART_0001": {
                "enabled": True,
                "marker_id": 123,
                "required_stable_frames": 5,
                "allowed_marker_area": {
                    "x_min": -0.60,
                    "x_max": 0.60,
                    "y_min": 0.20,
                    "y_max": 1.80,
                    "z_min": 0.20,
                    "z_max": 2.00,
                },
            }
        },
        "waiting_window": {
            "x_min": -0.35,
            "x_max": 0.35,
            "distance_min": 0.25,
            "distance_max": 0.70,
            "yaw_abs_max_deg": 10.0,
            "stable_marker_frames": 5,
            "stability_mode": "anchored_position",
            "position_center_speed_warn_px_s": 5.0,
        },
        "environment_profiles": {
            "indoor": {
                "waiting_window": {
                    "position_hold_time_sec": 0.8,
                    "position_anchor_drift_max_px": 6.0,
                    "position_history_radius_max_px": 2.5,
                }
            },
            "outdoor": {
                "waiting_window": {
                    "position_hold_time_sec": 1.0,
                    "position_anchor_drift_max_px": 8.0,
                    "position_history_radius_max_px": 3.0,
                }
            },
        },
        "safety": {
            "stop_zone": {
                "x_min": -0.50,
                "x_max": 0.50,
                "y_min": 0.20,
                "y_max": 1.80,
                "z_min": 0.00,
                "z_max": 0.80,
            },
            "warning_zone": {
                "x_min": -0.75,
                "x_max": 0.75,
                "y_min": 0.10,
                "y_max": 1.90,
                "z_min": 0.00,
                "z_max": 1.10,
            },
            "pir_stop_when_unknown_in_stop_zone": True,
        },
        "verified_cart_region": {
            "x_radius": 0.30,
            "y_radius": 0.45,
            "z_radius": 0.45,
        },
        "docking_corridor": {
            "x_min": -0.35,
            "x_max": 0.35,
            "distance_min": 0.25,
            "distance_max": 0.70,
            "yaw_abs_max_deg": 10.0,
        },
        "voice": {
            "allow_cart": "授权小车已停稳，允许接驳。",
            "wait_cart": "请等待小车停稳。",
            "stop_human": "检测到人员或肢体，请立即停止。",
            "stop_obstacle": "检测到障碍物，请立即停止。",
            "warning": "检测到接近风险，请注意。",
        },
    }
    cfg["waiting_window"].update(cfg["environment_profiles"][environment]["waiting_window"])
    return cfg

def marker(
    *,
    marker_id: int = 123,
    x: float = 0.0,
    z: float = 0.5,
    yaw: float = 0.0,
    frames: int = 8,
    center=(320.0, 240.0),
    center_speed: float = 0.0,
    jitter: float = 0.0,
) -> MarkerObservation:
    return MarkerObservation(
        marker_id=marker_id,
        pose=Pose(x=x, y=0.5, z=z, yaw_deg=yaw),
        stable_frames=frames,
        center_px=center,
        center_speed_px_s=center_speed,
        center_jitter_px=jitter,
        visible=True,
    )

def obj(label: str, x: float, z: float, *, confidence: float = 1.0, verified_cart: bool = False) -> ObjectObservation:
    return ObjectObservation(
        label=label,
        pose=Pose(x=x, y=0.4, z=z),
        confidence=confidence,
        belongs_to_verified_cart=verified_cart,
    )

def test_synthetic_acceptance_indoor_fixed_marker_allows_cart(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))

    first = engine.decide(SensorFrame(marker=marker(center=(320, 240), jitter=0.5)))
    assert first.decision == Decision.WARNING
    assert "position_hold_time" in first.reason

    now[0] = 101.0
    result = engine.decide(SensorFrame(marker=marker(center=(322, 240), jitter=1.0)))
    assert result.decision == Decision.ALLOW_CART
    assert "authorized_cart_in_waiting_window" in result.reason
    assert result.waiting_window and result.waiting_window.stable is True

def test_synthetic_acceptance_indoor_translation_waits(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240))))

    now[0] = 101.0
    result = engine.decide(SensorFrame(marker=marker(center=(328, 240), center_speed=8.0, jitter=1.0)))
    assert result.decision == Decision.WARNING
    assert "position_anchor_drift=8.0px>6.0px" in result.reason

def test_synthetic_acceptance_outdoor_light_sway_allows(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("outdoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240), jitter=0.5)))

    now[0] = 101.2
    result = engine.decide(SensorFrame(marker=marker(center=(326, 240), center_speed=12.0, jitter=2.5)))
    assert result.decision == Decision.ALLOW_CART
    assert result.waiting_window and "note=fast_local_sway" in result.waiting_window.reason

def test_synthetic_acceptance_marker_missing_idle():
    engine = FusionDecisionEngine(base_config("indoor"))
    result = engine.decide(SensorFrame(marker=None))
    assert result.decision == Decision.IDLE
    assert result.reason.startswith("cart_not_verified")

def test_synthetic_acceptance_wrong_marker_id_idle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    result = engine.decide(SensorFrame(marker=marker(marker_id=999)))
    assert result.decision == Decision.IDLE
    assert "cart_not_verified" in result.reason

def test_synthetic_acceptance_human_in_stop_zone_overrides_authorized_cart(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240))))
    now[0] = 101.0
    result = engine.decide(SensorFrame(marker=marker(center=(322, 240)), objects=[obj("human", 0.0, 0.4)]))
    assert result.decision == Decision.STOP
    assert "human" in result.reason

def test_synthetic_acceptance_unknown_obstacle_in_stop_zone_stops(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240))))
    now[0] = 101.0
    result = engine.decide(SensorFrame(marker=marker(center=(322, 240)), objects=[obj("box", 0.45, 0.5)]))
    assert result.decision == Decision.STOP
    assert "obstacle" in result.reason

def test_synthetic_acceptance_object_in_warning_zone_warns(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240))))
    now[0] = 101.0
    result = engine.decide(SensorFrame(marker=marker(center=(322, 240)), objects=[obj("box", 0.55, 0.5)]))
    assert result.decision == Decision.WARNING
    assert result.reason == "object_in_warning_zone"

def test_synthetic_acceptance_verified_cart_object_is_exempt_but_human_not(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(waiting_window_module.time, "time", lambda: now[0])
    engine = FusionDecisionEngine(base_config("indoor"))
    engine.decide(SensorFrame(marker=marker(center=(320, 240))))
    now[0] = 101.0
    result = engine.decide(
        SensorFrame(
            marker=marker(center=(322, 240)),
            objects=[obj("cart_body", 0.0, 0.4, verified_cart=True)],
        )
    )
    assert result.decision == Decision.ALLOW_CART

    result2 = engine.decide(
        SensorFrame(
            marker=marker(center=(322, 240)),
            objects=[obj("hand", 0.0, 0.4, verified_cart=True)],
        )
    )
    assert result2.decision == Decision.STOP
    assert "limb" in result2.reason or "human" in result2.reason
