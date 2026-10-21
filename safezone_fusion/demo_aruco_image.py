from __future__ import annotations

from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.adapters.mock_actuator import MockActuatorAdapter
from safezone_fusion.src.adapters.mock_business import BusinessEvent, MockBusinessAdapter
from safezone_fusion.src.config_loader import load_config
from safezone_fusion.src.interfaces import OperationMode, RuntimeInput
from safezone_fusion.src.orchestrator import SafeZoneOrchestrator
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.interfaces import CameraPacket, PirPacket, VisionSensorFusion
from safezone_fusion.src.vision.mock_pipeline import MockObjectDetector
from safezone_fusion.tools.generate_aruco_marker import generate

class ConstantPirReader:
    def read(self):
        return PirPacket(False)

def main():
    marker_path = ROOT / "generated_markers" / "aruco_DICT_6X6_250_123.png"
    if not marker_path.exists():
        generate(123, marker_path, "DICT_6X6_250", 800, 1)

    image = cv2.imread(str(marker_path))
    if image is None:
        raise RuntimeError(f"failed to read marker image: {marker_path}")

    config = load_config(ROOT / "config.yaml")
    orchestrator = SafeZoneOrchestrator(config)
    actuator = MockActuatorAdapter()

    detector = OpenCVArucoMarkerDetector(dictionary_name="DICT_6X6_250", marker_size_m=0.10, target_marker_ids={123})
    vision = VisionSensorFusion(
        marker_detector=detector,
        object_detector=MockObjectDetector([[], [], [], [], [], []]),
        pir_reader=ConstantPirReader(),
    )
    business = MockBusinessAdapter([BusinessEvent(OperationMode.CART_DOCKING, docking_requested=True) for _ in range(6)])

    print("SafeZone Fusion v0.6 ArUco image demo")
    print("marker image:", marker_path)
    print("=" * 60)

    for tick in range(6):
        sensors = vision.read(CameraPacket(left_rgb=image.copy(), timestamp=float(tick)))
        inp = RuntimeInput(business=business.read(), sensors=sensors, tick_id=tick)
        out = orchestrator.tick(inp)
        marker_info = None
        if sensors.marker:
            marker_info = f"id={sensors.marker.marker_id} stable={sensors.marker.stable_frames} pose=({sensors.marker.pose.x:.3f},{sensors.marker.pose.y:.3f},{sensors.marker.pose.z:.3f})"
        print(f"\n[tick {tick}] marker={marker_info}")
        actuator.apply(out.actuator)

if __name__ == "__main__":
    main()
