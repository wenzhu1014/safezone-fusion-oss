from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.adapters.mock_actuator import MockActuatorAdapter
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.adapters.mock_sensor import MockSensorAdapter, marker, obj
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.models import SensorFrame, SlotRequest
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator

def main():
    config = load_config(ROOT / "config.yaml")
    orchestrator = SafeZoneOrchestrator(config)

    slot_req = SlotRequest(
        slot_id="A-03",
        slot_x=0.10,
        slot_y=1.00,
        head_x=0.11,
        head_y=1.02,
        door_closed=True,
        user_done=False,
    )

    # Scripted timeline:
    # 0-1: cart docking, authorized cart, then hand intrusion stop.
    # 2-8: human access flow where slot is blocked and user steps back.
    events = [
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
        BusinessEvent(OperationMode.HUMAN_ACCESS, slot_request=slot_req),
    ]

    frames = [
        SensorFrame(marker=marker()),
        SensorFrame(marker=marker(), objects=[obj("hand", x=0.1, y=1.0, z=0.3)]),
        SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)]),
        SensorFrame(objects=[]),
        SensorFrame(objects=[]),
        SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)]),
        SensorFrame(objects=[obj("person", x=0.0, y=1.0, z=0.4)]),
        SensorFrame(objects=[]),
        SensorFrame(objects=[]),
    ]

    business = MockBusinessAdapter(events)
    sensors = MockSensorAdapter(frames)
    actuator = MockActuatorAdapter()

    print("SafeZone Fusion v0.4 mock runtime loop")
    print("=" * 60)
    for tick in range(len(events)):
        # Mutate the slot request across the scripted human access flow.
        if tick == 5:
            slot_req.door_closed = False  # user is accessing the slot
        if tick == 6:
            slot_req.door_closed = True
            slot_req.user_done = True

        inp = RuntimeInput(business=business.read(), sensors=sensors.read(), tick_id=tick)
        out = orchestrator.tick(inp)
        print(f"\n[tick {tick}] mode={inp.business.mode.value}")
        actuator.apply(out.actuator)

    print("\nHistory length:", len(actuator.history))

if __name__ == "__main__":
    main()
