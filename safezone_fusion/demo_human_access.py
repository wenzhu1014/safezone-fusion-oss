from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.human_access import HumanAccessController
from safezone_fusion.src.models import ObjectObservation, Pose, SensorFrame, SlotRequest

def obj(label, x=0.0, y=1.0, z=0.4, confidence=1.0):
    return ObjectObservation(label, Pose(x, y, z), confidence)

def print_result(step: str, result):
    print(f"\n{step}")
    print(f"  decision: {result.decision.value}")
    print(f"  reason:   {result.reason}")
    print(f"  voice:    {result.voice}")
    if result.debug:
        print(f"  debug:    {result.debug}")

def scenario_blocked_slot_user_steps_back():
    print("\n=== 场景 A：目标格口被设备挡住，用户先在危险区，后退后设备避让 ===")
    config = load_config(ROOT / "config.yaml")
    ctrl = HumanAccessController(config)

    req = SlotRequest(
        slot_id="A-03",
        slot_x=0.10,
        slot_y=1.00,
        head_x=0.11,
        head_y=1.02,
        door_closed=True,
        user_done=False,
    )

    # 1. 用户刚输入取餐码，人还在 stop_zone，设备不能动，只能提示后退。
    frame_user_near = SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)])
    print_result("1 用户在危险区，提示后退", ctrl.update(req, frame_user_near))

    # 2. 用户后退，安全区清空，设备可以移动到避让位。
    frame_clear = SensorFrame(objects=[])
    print_result("2 安全区清空，设备开始避让", ctrl.update(req, frame_clear))

    # 3. 仿真中下一帧认为设备已避让并锁定，允许用户靠近取餐。
    print_result("3 设备锁定，允许用户取餐", ctrl.update(req, frame_clear))

    # 4. 用户靠近取餐，此时设备静止锁定，不因为人进入而要求设备移动。
    req.door_closed = False
    print_result("4 用户靠近取餐，设备保持锁定", ctrl.update(req, frame_user_near))

    # 5. 用户完成并关门，但人还没离开，回位前提示后退。
    req.door_closed = True
    req.user_done = True
    print_result("5 用户完成但仍在危险区，回位前提示后退", ctrl.update(req, frame_user_near))

    # 6. 用户离开，设备回待机位。
    print_result("6 用户离开，设备开始复位", ctrl.update(req, frame_clear))

    # 7. 仿真中下一帧认为复位完成。
    print_result("7 复位完成", ctrl.update(req, frame_clear))

def scenario_slot_not_blocked():
    print("\n=== 场景 B：目标格口没被设备挡住，直接允许用户取餐 ===")
    config = load_config(ROOT / "config.yaml")
    ctrl = HumanAccessController(config)

    req = SlotRequest(
        slot_id="B-08",
        slot_x=-0.35,
        slot_y=1.40,
        head_x=0.20,
        head_y=0.80,
        door_closed=True,
        user_done=False,
    )
    frame_user_near = SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)])
    print_result("1 格口未被挡，设备不动并锁定，允许用户取餐", ctrl.update(req, frame_user_near))

if __name__ == "__main__":
    scenario_blocked_slot_user_steps_back()
    scenario_slot_not_blocked()
