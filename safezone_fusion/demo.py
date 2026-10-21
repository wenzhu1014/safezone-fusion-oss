from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.fusion import FusionDecisionEngine
from safezone_fusion.src.models import MarkerObservation, ObjectObservation, Pose, SensorFrame
from safezone_fusion.src.state_machine import SafetyStateMachine

def frame(marker=None, objects=None, pir=False):
    return SensorFrame(marker=marker, objects=objects or [], pir_triggered=pir)

def marker(mid=123, x=0.0, y=1.0, z=0.45, yaw=0.0, stable=6, speed=0.0, visible=True):
    return MarkerObservation(mid, Pose(x, y, z, yaw), stable, speed, visible)

def obj(label, x=0.0, y=1.0, z=0.4, confidence=1.0):
    return ObjectObservation(label, Pose(x, y, z), confidence)

def main():
    config = load_config(ROOT / "config.yaml")
    engine = FusionDecisionEngine(config)
    sm = SafetyStateMachine()

    scenarios = [
        ("01 授权小车停入等待窗口，安全区清空", frame(marker())),
        ("02 未授权小车", frame(marker(mid=9999))),
        ("03 授权码不稳定", frame(marker(stable=2))),
        ("04 授权小车但未停入等待窗口", frame(marker(x=0.8))),
        ("05 授权小车仍在移动，等待停稳", frame(marker(speed=0.05))),
        ("06 授权小车 + 人进入 stop_zone", frame(marker(), [obj("person", x=0.1, y=1.0, z=0.3)])),
        ("07 授权小车 + 手进入 stop_zone", frame(marker(), [obj("hand", x=0.1, y=1.0, z=0.3)])),
        ("08 授权小车本体在 marker 附近，不当未知障碍物", frame(marker(), [obj("unknown", x=0.2, y=1.0, z=0.5)])),
        ("09 授权小车 + 远离 marker 的未知障碍物进入 stop_zone", frame(marker(), [obj("unknown", x=0.49, y=1.7, z=0.3)])),
        ("10 授权码在授权区内但接驳走廊外，附近未知物不豁免", frame(marker(x=0.45), [obj("unknown", x=0.45, y=1.0, z=0.45)])),
        ("11 授权小车 + 手靠近 marker，人体/肢体永不豁免", frame(marker(), [obj("hand", x=0.05, y=1.0, z=0.45)])),
        ("12 预警区有人但未进入 stop_zone", frame(marker(), [obj("person", x=0.65, y=1.0, z=0.9)])),
        ("13 PIR 触发 + 未知障碍物进入 stop_zone", frame(marker(visible=False), [obj("unknown", x=0.0, y=1.0, z=0.4)], pir=True)),
        ("14 没有码，空闲等待", frame()),
    ]

    for name, f in scenarios:
        result = engine.decide(f, docking_requested=True)
        state = sm.update(result)
        cart_id = result.cart.cart_id if result.cart else None
        print(f"\n{name}")
        print(f"  decision: {result.decision.value}")
        print(f"  state:    {state.value}")
        print(f"  reason:   {result.reason}")
        print(f"  cart:     {cart_id} verified={result.cart.verified if result.cart else None}")
        print(f"  voice:    {result.voice}")

if __name__ == "__main__":
    main()
