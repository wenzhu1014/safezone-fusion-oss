from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.illumination import (
    IlluminationConfig,
    compensated_diff,
    estimate_global_shift,
    shadow_pixels,
    clean_mask,
)
from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector, RoiMotionConfig
from safezone_fusion.src.vision.interfaces import CameraPacket

# ---- unit: global shift ----

def test_estimate_global_shift():
    bg = np.full((20, 20), 100.0, dtype="float32")
    roi = bg + 30.0
    assert estimate_global_shift(roi, bg) == pytest.approx(30.0)

def test_compensated_diff_cancels_uniform_brightening():
    cfg = IlluminationConfig(compensate_global=True, max_global_shift=60)
    bg = np.full((20, 20), 100.0, dtype="float32")
    roi = bg + 40.0  # whole roi brighter (cloud moved / light)
    d = compensated_diff(roi, bg, cfg)
    # after compensation the diff should be ~0 everywhere
    assert float(d.max()) < 1.0

def test_compensated_diff_keeps_local_change():
    cfg = IlluminationConfig(compensate_global=True)
    bg = np.full((20, 20), 100.0, dtype="float32")
    roi = bg + 20.0
    roi[5:9, 5:9] = 230.0  # a local bright object on top of global shift
    d = compensated_diff(roi, bg, cfg)
    # the local patch must still stand out
    assert float(d[5:9, 5:9].mean()) > 80.0

# ---- unit: shadow ----

def test_shadow_pixels_detected():
    cfg = IlluminationConfig()
    bg = np.full((20, 20), 150.0, dtype="float32")
    roi = bg * 0.7  # uniform 30% darkening -> shadow
    sh = shadow_pixels(roi, bg, cfg)
    assert sh.mean() > 0.9  # almost all flagged shadow

def test_real_dark_object_not_shadow():
    cfg = IlluminationConfig()
    bg = np.full((20, 20), 150.0, dtype="float32")
    roi = bg.copy()
    roi[5:9, 5:9] = 10.0  # very dark occluding object (ratio ~0.07 < low band)
    sh = shadow_pixels(roi, bg, cfg)
    # the dark object patch should NOT be classified as shadow
    assert sh[5:9, 5:9].sum() == 0

def test_dark_clothed_person_over_textured_bg_not_shadow():
    """SAFETY: a person in dark clothing (40-60% brightness) over a TEXTURED
    surface must NOT be suppressed as shadow -- texture is replaced, not kept."""
    rng = np.random.default_rng(5)
    bg = rng.integers(80, 220, size=(40, 40)).astype("float32")  # textured floor
    roi = bg.copy()
    # dark flat clothing patch: ~50% brightness, FLAT (texture replaced)
    roi[10:30, 10:30] = 90.0
    cfg = IlluminationConfig()
    sh = shadow_pixels(roi, bg, cfg)
    # the clothing patch must mostly survive (not suppressed as shadow)
    patch = sh[12:28, 12:28]
    assert patch.mean() < 0.3, "dark flat object over texture must not be called shadow"

def test_shadow_over_textured_bg_is_suppressed():
    """A shadow over a textured surface keeps texture -> should be suppressed."""
    rng = np.random.default_rng(6)
    bg = rng.integers(80, 220, size=(40, 40)).astype("float32")
    roi = bg.copy()
    roi[10:30, 10:30] = bg[10:30, 10:30] * 0.7  # shadow: scaled, texture preserved
    cfg = IlluminationConfig()
    sh = shadow_pixels(roi, bg, cfg)
    patch = sh[12:28, 12:28]
    assert patch.mean() > 0.6, "shadow over texture should be largely suppressed"

# ---- integration with the detector ----

def _det():
    cfg = RoiMotionConfig(enabled=True, roi=(0.0, 0.0, 1.0, 1.0), min_area_px=300.0,
                          warmup_frames=3, min_consecutive=2, threshold=25,
                          marker_mask_pad_px=-1)
    return RoiMotionObjectDetector(cfg)

def _frame(val, h=240, w=320):
    return np.full((h, w, 3), val, dtype=np.uint8)

def _frame_textured(base, h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((h, w, 3), base, dtype=np.uint8)
    noise = rng.integers(-15, 15, size=(h, w, 3))
    return np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8)

def test_global_brightness_change_does_not_trigger():
    det = _det()
    # learn background at brightness 90
    for _ in range(5):
        det.detect(CameraPacket(rgb=_frame_textured(90, seed=1)))
    # whole scene brightens to 140 (cloud passes) - same texture, +50 global
    triggered = False
    for _ in range(4):
        det.detect(CameraPacket(rgb=_frame_textured(140, seed=1)))
        triggered = triggered or det.last_triggered
    assert triggered is False, "uniform brightness change must not trigger STOP"

def test_real_object_still_triggers_under_illumination_robustness():
    det = _det()
    for _ in range(5):
        det.detect(CameraPacket(rgb=_frame_textured(90, seed=2)))
    # a bright object appears in the ROI
    triggered = False
    for _ in range(4):
        f = _frame_textured(90, seed=2)
        f[100:160, 120:200] = 230
        det.detect(CameraPacket(rgb=f))
        triggered = triggered or det.last_triggered
    assert triggered is True, "a real object must still trigger STOP"

def test_shadow_does_not_trigger():
    det = _det()
    for _ in range(5):
        det.detect(CameraPacket(rgb=_frame_textured(160, seed=3)))
    # a shadow falls on part of the ROI: that region darkens ~30%, texture kept
    triggered = False
    for _ in range(4):
        f = _frame_textured(160, seed=3).astype("float32")
        f[80:180, 100:220] *= 0.7  # cast shadow
        det.detect(CameraPacket(rgb=f.astype(np.uint8)))
        triggered = triggered or det.last_triggered
    assert triggered is False, "a cast shadow must not trigger STOP"
