from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.camera_autodetect import (
    CapturePlan,
    NoCameraError,
    plan_from_probes,
)

@dataclass
class FakeProbe:
    index: int
    opened: bool
    role_guess: str = "unknown"

def test_no_camera_raises():
    with pytest.raises(NoCameraError):
        plan_from_probes([FakeProbe(0, False), FakeProbe(1, False)])

def test_single_rgb():
    plan = plan_from_probes([FakeProbe(0, True, "rgb")])
    assert plan.primary_index == 0
    assert plan.primary_role == "rgb"
    assert plan.secondary_index is None

def test_single_nir():
    plan = plan_from_probes([FakeProbe(2, True, "mono_or_nir")])
    assert plan.primary_index == 2
    assert plan.primary_role == "nir"

def test_single_unknown_assumes_rgb():
    plan = plan_from_probes([FakeProbe(0, True, "unknown")])
    assert plan.primary_role == "rgb"
    assert "unknown" in plan.note

def test_dual_rgb_and_nir_assigns_rgb_primary():
    # NIR on lower index, RGB on higher: RGB must still become primary
    plan = plan_from_probes([FakeProbe(0, True, "mono_or_nir"), FakeProbe(1, True, "rgb")])
    assert plan.primary_index == 1
    assert plan.primary_role == "rgb"
    assert plan.secondary_index == 0
    assert plan.secondary_role == "nir"

def test_dual_picks_lowest_index_per_role():
    probes = [FakeProbe(0, True, "rgb"), FakeProbe(1, True, "rgb"),
              FakeProbe(2, True, "mono_or_nir")]
    plan = plan_from_probes(probes)
    assert plan.primary_index == 0
    assert plan.primary_role == "rgb"
    assert plan.secondary_index == 2
    assert plan.secondary_role == "nir"

def test_both_mono():
    plan = plan_from_probes([FakeProbe(0, True, "mono_or_nir"), FakeProbe(1, True, "mono_or_nir")])
    assert plan.primary_role == "nir"
    assert plan.secondary_index == 1

def test_skips_unopened():
    plan = plan_from_probes([FakeProbe(0, False), FakeProbe(1, True, "rgb")])
    assert plan.primary_index == 1

def test_to_cli_args_single():
    plan = CapturePlan(primary_index=0, primary_role="rgb")
    assert plan.to_cli_args() == ["--camera-index", "0", "--primary-role", "rgb"]

def test_to_cli_args_dual():
    plan = CapturePlan(primary_index=1, primary_role="rgb",
                       secondary_index=0, secondary_role="nir")
    args = plan.to_cli_args()
    assert args == ["--camera-index", "1", "--primary-role", "rgb",
                    "--secondary-index", "0", "--secondary-role", "nir"]
