from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import BusinessInput, OperationMode, RuntimeInput
from safezone_fusion.src.models import MarkerObservation, ObjectObservation, Pose, SensorFrame, SlotRequest
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator

def marker(mid=123, x=0.0, y=1.0, z=0.45, yaw=0.0, stable=6, speed=0.0, visible=True):
    return MarkerObservation(mid, Pose(x, y, z, yaw), stable, speed, visible)

def obj(label, x=0.0, y=1.0, z=0.4, confidence=1.0):
    return ObjectObservation(label, Pose(x, y, z), confidence)

def show(title: str, out):
    a = out.actuator
    print(f"\n{title}")
    print(f"  mode:      {out.mode.value}")
    print(f"  decision:  {a.raw_decision}")
    print(f"  motor:     {a.motor.value}")
    print(f"  door:      {a.door.value}")
    print(f"  reason:    {a.reason}")
    print(f"  voice:     {a.voice}")
    if a.debug:
        print(f"  debug:     {a.debug}")

def main():
    config = load_config(ROOT / "config.yaml")
    orch = SafeZoneOrchestrator(config)

    # A. Cart docking: authorized cart in waiting window -> allow docking.
    cart_inp = RuntimeInput(
        business=BusinessInput(mode=OperationMode.CART_DOCKING, docking_requested=True),
        sensors=SensorFrame(marker=marker()),
        tick_id=1,
    )
    show("A1 授权小车停入等待窗口", orch.tick(cart_inp))

    # B. Cart docking: human enters stop zone -> stop even with authorized cart.
    cart_stop_inp = RuntimeInput(
        business=BusinessInput(mode=OperationMode.CART_DOCKING, docking_requested=True),
        sensors=SensorFrame(marker=marker(), objects=[obj("hand", x=0.1, y=1.0, z=0.3)]),
        tick_id=2,
    )
    show("A2 授权小车 + 手进入危险区", orch.tick(cart_stop_inp))

    # C. Human access: slot is blocked, user near -> ask clear.
    slot_req = SlotRequest(
        slot_id="A-03",
        slot_x=0.10,
        slot_y=1.00,
        head_x=0.11,
        head_y=1.02,
        door_closed=True,
        user_done=False,
    )
    human_near = SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)])
    human_clear = SensorFrame(objects=[])
    human_business = BusinessInput(mode=OperationMode.HUMAN_ACCESS, slot_request=slot_req)

    show("B1 人工取餐：格口被挡且用户在危险区", orch.tick(RuntimeInput(human_business, human_near, 3)))
    show("B2 用户后退，设备避让", orch.tick(RuntimeInput(human_business, human_clear, 4)))
    show("B3 设备锁定，开格口", orch.tick(RuntimeInput(human_business, human_clear, 5)))

    # User accesses slot; device stays locked.
    slot_req.door_closed = False
    show("B4 用户靠近取餐，设备锁定", orch.tick(RuntimeInput(human_business, human_near, 6)))

    # User done; ask clear before return.
    slot_req.door_closed = True
    slot_req.user_done = True
    show("B5 用户完成但仍在危险区，复位前提示", orch.tick(RuntimeInput(human_business, human_near, 7)))
    show("B6 用户离开，设备复位", orch.tick(RuntimeInput(human_business, human_clear, 8)))
    show("B7 复位完成", orch.tick(RuntimeInput(human_business, human_clear, 9)))

if __name__ == "__main__":
    main()
