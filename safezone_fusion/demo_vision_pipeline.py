from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.adapters.mock_actuator import MockActuatorAdapter
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.vision.interfaces import CameraPacket, VisionSensorFusion
from safezone_fusion.src.vision.mock_pipeline import MockMarkerDetector, MockObjectDetector, MockPirReader, marker, obj

def main():
    config = load_config(ROOT / "config.yaml")
    orchestrator = SafeZoneOrchestrator(config)
    actuator = MockActuatorAdapter()

    business = MockBusinessAdapter([
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
        BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True),
    ])

    vision = VisionSensorFusion(
        marker_detector=MockMarkerDetector([
            None,                  # tick0: no marker
            marker(stable=2),       # tick1: candidate but not stable
            marker(stable=6),       # tick2: authorized and stable
            marker(stable=6),       # tick3: authorized but hand intrusion
        ]),
        object_detector=MockObjectDetector([
            [],
            [],
            [],
            [obj("hand", x=0.1, y=1.0, z=0.3)],
        ]),
        pir_reader=MockPirReader([False, False, False, True]),
    )

    print("SafeZone Fusion v0.5 mock vision pipeline")
    print("=" * 60)
    for tick in range(4):
        packet = CameraPacket(timestamp=float(tick), meta={"source": "mock-dual-stereo"})
        sensors = vision.read(packet)
        inp = RuntimeInput(business=business.read(), sensors=sensors, tick_id=tick)
        out = orchestrator.tick(inp)
        print(f"\n[tick {tick}] marker={sensors.marker.marker_id if sensors.marker else None} objects={[o.label for o in sensors.objects]} pir={sensors.pir_triggered}")
        actuator.apply(out.actuator)

if __name__ == "__main__":
    main()
