from __future__ import annotations

"""Following danger zone: a danger region that tracks the moving device.

The problem (from 博)
---------------------
The docking device (a "backpack" on the cabinet face) can move across the WHOLE
face. If we drew one static danger box big enough to cover everywhere it can go,
the box would be the entire face and safety detection would be meaningless.

The insight
-----------
At any instant the device occupies only a SMALL patch of the face. The pinch
hazard exists only *around the device body that is currently moving* -- not at
some far corner of the face where there is no mechanism. So the danger zone
should FOLLOW the device, not cover the whole face.

The PLC knows where the device is (it drives the motors). So:

    PLC device pose --> map to pixels --> danger zone = a box around the device
                                          (device body excluded, surrounding ring
                                           is where we look for a hand/limb)

As the device moves, this small box moves with it. We never watch the whole
face at once; we watch the moving neighbourhood of the device.

This module is pure geometry (normalized + pixel boxes). The normalized->pixel
mapping is a calibration (drive device to the 4 corners, record pixels, fit);
here we accept an affine/linear map so the logic is testable without hardware.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

Box = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def _clamp_box01(box: Box) -> Box:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((_clamp01(x0), _clamp01(x1)))
    y0, y1 = sorted((_clamp01(y0), _clamp01(y1)))
    return (x0, y0, x1, y1)

@dataclass
class FollowingZoneConfig:
    """Geometry of the device-following danger zone, in normalized face coords.

    All sizes are fractions of the cabinet-face image (0..1).

    - device_half_w/h: half-size of the device BODY footprint (excluded region).
      Things inside this are the device itself -> not an obstacle.
    - margin: how far OUT from the device body we still consider dangerous
      (the ring where a hand could get pinched). The danger zone is the device
      body grown by this margin; the body itself is cut out.
    - inflate_when_moving: extra margin added while the device is actively
      moving (a moving mechanism needs a bit more clearance). 0 = no change.
    - whole_face_when_unknown: if device pose is unknown, fall back to watching
      a default box (safety-biased) rather than nothing.
    """

    device_half_w: float = 0.08
    device_half_h: float = 0.08
    margin: float = 0.06
    inflate_when_moving: float = 0.03
    whole_face_when_unknown: bool = True
    fallback_box: Box = (0.3, 0.5, 0.7, 0.95)

@dataclass
class FollowingZoneResult:
    danger_box: Optional[Box]      # outer danger region (normalized) to watch
    device_box: Optional[Box]      # device body region (normalized) to EXCLUDE
    known: bool
    reason: str = ""

def compute_following_zone(
    device_x: Optional[float],
    device_y: Optional[float],
    moving: bool,
    cfg: FollowingZoneConfig,
) -> FollowingZoneResult:
    """Build the danger box (to watch) and device box (to exclude) from pose.

    Both boxes are normalized (0..1) in the cabinet-face frame. The detector
    should look for a hand/limb inside ``danger_box`` but ignore anything inside
    ``device_box`` (that is the device itself).
    """
    if device_x is None or device_y is None:
        if cfg.whole_face_when_unknown:
            return FollowingZoneResult(
                danger_box=_clamp_box01(cfg.fallback_box),
                device_box=None,
                known=False,
                reason="device_pose_unknown:fallback_box",
            )
        return FollowingZoneResult(None, None, False, "device_pose_unknown:no_zone")

    cx, cy = _clamp01(device_x), _clamp01(device_y)
    extra = cfg.inflate_when_moving if moving else 0.0

    dev_hw, dev_hh = cfg.device_half_w, cfg.device_half_h
    device_box = _clamp_box01((cx - dev_hw, cy - dev_hh, cx + dev_hw, cy + dev_hh))

    out_hw = dev_hw + cfg.margin + extra
    out_hh = dev_hh + cfg.margin + extra
    danger_box = _clamp_box01((cx - out_hw, cy - out_hh, cx + out_hw, cy + out_hh))

    reason = "following_zone:moving" if moving else "following_zone:static"
    return FollowingZoneResult(danger_box=danger_box, device_box=device_box,
                               known=True, reason=reason)

@dataclass
class FaceToPixelCalibration:
    """Affine map from normalized face coords (0..1) to image pixels.

    Calibration procedure (on-site, once): drive the device to known points on
    the face, record the device center pixel for each, fit this map. For a
    fronto-parallel ceiling camera a simple linear scale+offset is enough:

        px = x * sx + ox
        py = y * sy + oy

    Defaults map the unit square onto a full-frame box, i.e. identity over the
    given width/height, which is a reasonable starting point before calibration.
    """

    sx: float
    sy: float
    ox: float = 0.0
    oy: float = 0.0

    @classmethod
    def identity(cls, width: int, height: int) -> "FaceToPixelCalibration":
        return cls(sx=float(width), sy=float(height), ox=0.0, oy=0.0)

    def box_to_pixels(self, box: Optional[Box], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        if box is None:
            return None
        x0, y0, x1, y1 = box
        px0 = int(round(x0 * self.sx + self.ox))
        py0 = int(round(y0 * self.sy + self.oy))
        px1 = int(round(x1 * self.sx + self.ox))
        py1 = int(round(y1 * self.sy + self.oy))
        px0, px1 = sorted((max(0, min(width, px0)), max(0, min(width, px1))))
        py0, py1 = sorted((max(0, min(height, py0)), max(0, min(height, py1))))
        return (px0, py0, px1, py1)
