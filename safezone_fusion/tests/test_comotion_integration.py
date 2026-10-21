from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.vision.roi_motion_detector import RoiMotionObjectDetector, RoiMotionConfig
from safezone_fusion.src.vision.interfaces import CameraPacket

W, H = 320, 240

def _blank():
    return np.full((H, W, 3), 40, dtype=np.uint8)

def _block(frame, cx, cy, half=22, value=210):
    f = frame.copy()
    x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
    y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
    f[y0:y1, x0:x1] = value
    return f

def _make_detector():
    cfg = RoiMotionConfig(enabled=True, roi=(0.0, 0.0, 1.0, 1.0), min_area_px=300.0,
                          warmup_frames=3, min_consecutive=2, threshold=25,
                          marker_mask_pad_px=-1)  # disable marker box mask to isolate co-motion
    return RoiMotionObjectDetector(cfg)

def test_cart_body_co_moving_with_marker_is_exempt():
    """博's scenario: cart body slides in together with its authorized marker.
    Co-motion should exempt it -> no STOP."""
    det = _make_detector()
    for _ in range(4):
        det.detect(CameraPacket(rgb=_blank()))  # warmup empty

    triggered_any = False
    # cart body + marker slide right together across the face
    for i in range(8):
        cx = 80 + i * 12
        cy = 130
        # marker rides on the body (slightly above center), same motion
        det.set_marker_comotion((cx, cy - 10), exempt_active=True)
        frame = _block(_blank(), cx, cy, half=26)
        det.detect(CameraPacket(rgb=frame))
        triggered_any = triggered_any or det.last_triggered
    assert triggered_any is False, "cart body moving with marker must be exempt"

def test_hand_from_different_direction_still_stops():
    """A hand entering from a different direction does NOT move with the marker
    -> must STOP even though an authorized marker is present and moving."""
    det = _make_detector()
    for _ in range(4):
        det.detect(CameraPacket(rgb=_blank()))

    # marker moving right slowly at top; a hand drops straight down elsewhere
    triggered = False
    for i in range(8):
        mx = 80 + i * 12
        det.set_marker_comotion((mx, 60), exempt_active=True)
        # hand moves vertically (downward) on the right side -> different direction
        hy = 60 + i * 12
        frame = _block(_blank(), 240, hy, half=24)
        det.detect(CameraPacket(rgb=frame))
        triggered = triggered or det.last_triggered
    assert triggered is True, "hand from another direction must still trigger STOP"

def test_no_exemption_when_marker_not_authorized():
    """exempt_active=False (marker not authorized / not in corridor): even a
    co-moving blob must NOT be exempted."""
    det = _make_detector()
    for _ in range(4):
        det.detect(CameraPacket(rgb=_blank()))

    triggered = False
    for i in range(8):
        cx = 80 + i * 12
        det.set_marker_comotion((cx, 120), exempt_active=False)  # not authorized
        frame = _block(_blank(), cx, 130, half=26)
        det.detect(CameraPacket(rgb=frame))
        triggered = triggered or det.last_triggered
    assert triggered is True, "no exemption allowed when marker not authorized/in-corridor"

def test_stopped_cart_not_exempted_by_comotion():
    """When the marker is stationary, co-motion is undecidable -> co-motion does
    NOT exempt (a settled cart must be handled by position-based exemption,
    which is the semantic layer's job, not roi_motion's)."""
    det = _make_detector()
    for _ in range(4):
        det.detect(CameraPacket(rgb=_blank()))

    triggered = False
    for _ in range(6):
        det.set_marker_comotion((150, 120), exempt_active=True)  # marker not moving
        frame = _block(_blank(), 150, 130, half=26)
        det.detect(CameraPacket(rgb=frame))
        triggered = triggered or det.last_triggered
    assert triggered is True, "stopped cart is not exempted by co-motion (undecidable)"
