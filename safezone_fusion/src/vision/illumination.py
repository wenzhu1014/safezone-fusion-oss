from __future__ import annotations

"""Illumination robustness + shadow suppression for occupancy detection.

Why (博, 2026-06-22; indoor & outdoor cabinets)
-----------------------------------------------
Background-difference occupancy is sensitive to two very common real-world
effects that are NOT obstacles:

1. Global illumination change. A cloud passing, a light switching, dusk/dawn,
   a car headlight sweeping -> the WHOLE roi gets brighter or darker fairly
   uniformly. Raw absdiff then lights up everywhere and causes false STOPs.
   Fix: estimate the uniform brightness shift between roi and background and
   compensate it before thresholding, so only LOCAL changes survive.

2. Cast shadows. A person/cloud shadow moving across the cabinet face is a
   moving dark region -> classic false obstacle. A shadow makes pixels DARKER
   while leaving the underlying texture/structure essentially unchanged.
   Fix: detect pixels that are (a) darker than background and (b) a roughly
   uniform multiplicative darkening (ratio in a shadow band), and remove them
   from the obstacle mask.

Both are conservative and safety-biased: we only suppress what clearly looks
like illumination/shadow, and a real object (which changes texture and is not a
uniform darkening) still survives and triggers STOP.

Pure numpy so it is unit-testable without a camera; cv2 not required here.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class IlluminationConfig:
    enabled: bool = True
    # Global brightness compensation: subtract the robust (median) brightness
    # shift between roi and bg before differencing. Cap the compensation so a
    # huge real change is not fully cancelled.
    compensate_global: bool = True
    max_global_shift: float = 60.0
    # Shadow suppression: a pixel is shadow-like if it is darker than bg and the
    # ratio roi/bg falls in [shadow_ratio_low, shadow_ratio_high] (uniform
    # darkening). Highlights (over-bright reflections) handled by ratio>high.
    suppress_shadow: bool = True
    shadow_ratio_low: float = 0.35
    shadow_ratio_high: float = 0.92
    # Only treat as shadow when darker by at least this much (avoid nuking
    # near-zero noise which threshold handles anyway).
    shadow_min_darken: float = 8.0
    # SAFETY: require local texture preserved to call something a shadow (a dark
    # object replaces texture; a shadow keeps it). Without this, a person in
    # dark clothing could be mis-suppressed as shadow -> missed detection.
    require_texture_preserved: bool = True
    texture_flat_thresh: float = 6.0     # bg gradient below this = flat (no texture to lose)
    texture_keep_ratio: float = 0.45     # roi must keep >=45% of bg's local texture

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IlluminationConfig":
        base = cls()
        if not data:
            return base
        return cls(
            enabled=bool(data.get("enabled", base.enabled)),
            compensate_global=bool(data.get("compensate_global", base.compensate_global)),
            max_global_shift=float(data.get("max_global_shift", base.max_global_shift)),
            suppress_shadow=bool(data.get("suppress_shadow", base.suppress_shadow)),
            shadow_ratio_low=float(data.get("shadow_ratio_low", base.shadow_ratio_low)),
            shadow_ratio_high=float(data.get("shadow_ratio_high", base.shadow_ratio_high)),
            shadow_min_darken=float(data.get("shadow_min_darken", base.shadow_min_darken)),
            require_texture_preserved=bool(data.get("require_texture_preserved", base.require_texture_preserved)),
            texture_flat_thresh=float(data.get("texture_flat_thresh", base.texture_flat_thresh)),
            texture_keep_ratio=float(data.get("texture_keep_ratio", base.texture_keep_ratio)),
        )

def estimate_global_shift(roi, bg):
    """Robust estimate of the uniform brightness shift (roi - bg), via median."""
    import numpy as np
    r = np.asarray(roi, dtype="float32")
    b = np.asarray(bg, dtype="float32")
    return float(np.median(r - b))

def compensated_diff(roi, bg, cfg: IlluminationConfig):
    """Absolute difference after removing the uniform brightness shift.

    Returns a float32 array of the same shape. When compensation is disabled,
    this is just |roi - bg|.
    """
    import numpy as np
    r = np.asarray(roi, dtype="float32")
    b = np.asarray(bg, dtype="float32")
    if cfg.compensate_global:
        shift = estimate_global_shift(r, b)
        # cap so a genuine large change is not fully cancelled
        if shift > cfg.max_global_shift:
            shift = cfg.max_global_shift
        elif shift < -cfg.max_global_shift:
            shift = -cfg.max_global_shift
        b = b + shift
    return np.abs(r - b)

def shadow_pixels(roi, bg, cfg: IlluminationConfig):
    """Boolean mask of pixels that look like a cast shadow (uniform darkening).

    SAFETY-CRITICAL discriminator: brightness alone CANNOT separate a shadow
    from a dark object (a person in dark clothing can be 40-60% of background
    brightness, same as a shadow). The real difference is TEXTURE:

      * a shadow darkens the surface but the underlying texture/structure shows
        through (gradients are preserved, just scaled down) ->  suppress
      * a real object REPLACES the surface, so the local texture/gradient does
        not match the background ->  keep (STOP)

    So a pixel is shadow only if (a) it is a uniform darkening in the shadow
    ratio band AND (b) the local texture is preserved vs background. When in
    doubt we keep the pixel (fail toward STOP).
    """
    import numpy as np
    r = np.asarray(roi, dtype="float32")
    b = np.asarray(bg, dtype="float32")
    darker = (b - r) >= cfg.shadow_min_darken
    safe_b = np.maximum(b, 1.0)
    ratio = r / safe_b
    in_band = (ratio >= cfg.shadow_ratio_low) & (ratio <= cfg.shadow_ratio_high)
    candidate = darker & in_band
    if not cfg.require_texture_preserved:
        return candidate
    # Texture-preservation gate: compare local gradient magnitude. A shadow
    # keeps relative texture; a flat/foreign object does not.
    tex_ok = _texture_preserved(r, b, cfg)
    return candidate & tex_ok

def _texture_preserved(r, b, cfg: "IlluminationConfig"):
    """Per-pixel bool: is local texture preserved between roi r and bg b?

    Uses gradient magnitude as a cheap texture proxy. In a shadow the surface
    texture persists, so bg has texture wherever roi has texture. In a flat dark
    object the roi loses the background's texture. We flag 'preserved' where the
    background is itself fairly flat (no texture to lose -> brightness test is
    enough) OR the roi retains a meaningful fraction of the background texture.
    """
    import numpy as np
    def grad(a):
        gx = np.zeros_like(a); gy = np.zeros_like(a)
        gx[:, 1:] = np.abs(a[:, 1:] - a[:, :-1])
        gy[1:, :] = np.abs(a[1:, :] - a[:-1, :])
        return gx + gy
    gr = grad(r)
    gb = grad(b)
    flat_bg = gb <= cfg.texture_flat_thresh          # bg has little texture here
    # roi keeps at least `texture_keep_ratio` of bg's local texture
    keeps = gr >= (cfg.texture_keep_ratio * gb)
    return flat_bg | keeps

def clean_mask(mask, roi, bg, cfg: IlluminationConfig):
    """Remove shadow pixels from a binary obstacle mask.

    ``mask`` is a uint8 0/255 array; returns the same with shadow pixels zeroed.
    """
    import numpy as np
    if not cfg.enabled or not cfg.suppress_shadow:
        return mask
    sh = shadow_pixels(roi, bg, cfg)
    out = mask.copy()
    out[sh] = 0
    return out
