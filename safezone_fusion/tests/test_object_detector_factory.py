from __future__ import annotations

import numpy as np
import pytest

from safezone_fusion.src.vision.interfaces import CameraPacket
from safezone_fusion.src.vision.mock_pipeline import MockObjectDetector
from safezone_fusion.src.vision.object_detector_factory import (
    available_object_detectors,
    build_object_detector,
    register_object_detector,
    resolve_object_detector,
)
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector
from safezone_fusion.src.vision.skin_motion_detector import SkinMotionObjectDetector

def test_factory_default_mock():
    name, detector = resolve_object_detector({})
    assert name == "mock"
    assert isinstance(detector, MockObjectDetector)

def test_factory_uses_roi_motion_when_enabled_in_config():
    config = {"roi_motion": {"enabled": True, "roi": [0.1, 0.1, 0.9, 0.9]}}
    name, detector = resolve_object_detector(config)
    assert name == "roi_motion"
    assert isinstance(detector, RoiMotionObjectDetector)
    assert detector.config.roi == (0.1, 0.1, 0.9, 0.9)

def test_factory_force_roi_motion_via_flag():
    config = {"roi_motion": {"enabled": False, "roi": [0.2, 0.2, 0.8, 0.8]}}
    name, detector = resolve_object_detector(config, roi_motion_force_enabled=True)
    assert name == "roi_motion"
    assert isinstance(detector, RoiMotionObjectDetector)

def test_factory_force_roi_motion_overrides_explicit_name():
    config = {
        "object_detector": {"name": "mock"},
        "roi_motion": {"enabled": False, "roi": [0.2, 0.2, 0.8, 0.8]},
    }
    name, detector = resolve_object_detector(config, roi_motion_force_enabled=True)
    assert name == "roi_motion"
    assert isinstance(detector, RoiMotionObjectDetector)

def test_factory_explicit_name_overrides_roi_motion_block():
    config = {
        "object_detector": {"name": "mock"},
        "roi_motion": {"enabled": True, "roi": [0.0, 0.0, 1.0, 1.0]},
    }
    name, detector = resolve_object_detector(config)
    assert name == "mock"
    assert isinstance(detector, MockObjectDetector)

def test_register_custom_detector_and_resolve():
    class _Probe:
        def __init__(self, cfg):
            self.cfg = dict(cfg)

        def detect(self, _packet):
            return []

    register_object_detector("probe_only", lambda cfg: _Probe(cfg))
    assert "probe_only" in available_object_detectors()

    config = {"object_detector": {"name": "probe_only", "config": {"threshold": 5}}}
    name, detector = resolve_object_detector(config)
    assert name == "probe_only"
    assert isinstance(detector, _Probe)
    assert detector.cfg == {"threshold": 5}

def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build_object_detector("does_not_exist", {})

def test_factory_skin_motion_backend():
    cfg = {"object_detector": {"name": "skin_motion"}, "skin_motion": {"enabled": False, "roi": [0.1, 0.2, 0.3, 0.4]}}
    name, detector = resolve_object_detector(cfg)
    assert name == "skin_motion"
    assert isinstance(detector, SkinMotionObjectDetector)
    assert detector.config.roi == (0.1, 0.2, 0.3, 0.4)
    assert "mediapipe_hand" in available_object_detectors()

def test_mediapipe_missing_dependency_is_loud_on_first_detection(monkeypatch, tmp_path):
    """Construction stays lazy; a missing optional dependency fails at use."""
    import builtins
    model = tmp_path / "example.task"
    model.write_bytes(b"not a real model; imports must fail before model loading")
    original_import = builtins.__import__

    def without_mediapipe(name, *args, **kwargs):
        if name == "mediapipe" or name.startswith("mediapipe."):
            raise ImportError("optional dependency deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mediapipe)
    det = build_object_detector("mediapipe_hand", {"model_path": str(model)})
    packet = CameraPacket(rgb=np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="optional mediapipe"):
        det.detect(packet)


def test_mediapipe_requires_explicit_model_before_detection():
    det = build_object_detector("mediapipe_hand", {})
    packet = CameraPacket(rgb=np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="local model_path"):
        det.detect(packet)


def test_mediapipe_disabled_or_empty_packet_does_not_initialize(monkeypatch):
    det = build_object_detector("mediapipe_hand", {"enabled": False})

    def unexpected_init():
        pytest.fail("disabled or empty input must not load a model")

    monkeypatch.setattr(det, "_lazy_init", unexpected_init)
    assert det.detect(CameraPacket(rgb=np.zeros((16, 16, 3), dtype=np.uint8))) == []
    det.config.enabled = True
    assert det.detect(CameraPacket()) == []

def test_resolve_object_detectors_force_enables_despite_config_false():
    """Regression (2026-06-22): config blocks ship enabled:false; selecting a
    detector by name must force-enable it, not leave the False in place.
    setdefault was the bug -- it left enabled:False and silently disabled both
    detectors so nothing was ever detected in layered mode."""
    from safezone_fusion.src.vision.object_detector_factory import resolve_object_detectors
    cfg = {
        "mediapipe_hand": {"enabled": False, "roi": [0.3, 0.62, 0.7, 0.95]},
        "roi_motion": {"enabled": False, "roi": [0.3, 0.62, 0.7, 0.95]},
    }
    names, dets = resolve_object_detectors(cfg, ["mediapipe_hand", "roi_motion"])
    assert names == ["mediapipe_hand", "roi_motion"]
    for d in dets:
        assert d.config.enabled is True
