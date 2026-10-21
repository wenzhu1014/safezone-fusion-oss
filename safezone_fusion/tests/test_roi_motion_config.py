from __future__ import annotations

import argparse
from pathlib import Path

from safezone_fusion.demo_live_camera import build_roi_motion_config
from safezone_fusion.src.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]

def test_roi_motion_config_defaults_from_yaml():
    config = load_config(ROOT / "config.yaml")
    args = argparse.Namespace(roi_motion=True, roi=None, roi_min_area=None, roi_threshold=None)
    out = build_roi_motion_config(config, args)
    assert out["enabled"] is True
    assert out["roi"] == [0.30, 0.62, 0.70, 0.95]
    assert float(out["min_area_px"]) == 1200.0
    assert int(out["threshold"]) == 28

def test_roi_motion_config_cli_overrides_yaml():
    config = load_config(ROOT / "config.yaml")
    args = argparse.Namespace(roi_motion=True, roi=[0.1, 0.2, 0.3, 0.4], roi_min_area=333.0, roi_threshold=11)
    out = build_roi_motion_config(config, args)
    assert out["enabled"] is True
    assert out["roi"] == [0.1, 0.2, 0.3, 0.4]
    assert out["min_area_px"] == 333.0
    assert out["threshold"] == 11
