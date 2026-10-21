"""Product-grade HUD overlay for SafeZone live view.

The existing draw_marker_debug() is an engineering/diagnostic view (many small
lines). This module adds a PRODUCT overlay meant for the cabinet screen / an
investor demo: a big colour-coded status banner in Chinese plus a dynamic
safety-zone highlight, readable at a glance from across a room.

OpenCV's putText cannot render CJK, so we draw text with PIL (a CJK font is
available on Windows: msyh.ttc) and composite back onto the BGR frame.

Pure and import-safe: rendering takes a frame + a HudState and returns a frame.
No camera, no global state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

# Status tiers, most -> least severe. Drives banner text + colour.
STATUS_STOP = "STOP"
STATUS_WARNING = "WARNING"
STATUS_ALLOW = "ALLOW"
STATUS_IDLE = "IDLE"
STATUS_SCANNING = "SCANNING"   # marker visible but still verifying/stabilising the cart

# BGR colours (OpenCV order).
_COLORS = {
    STATUS_STOP: (60, 60, 230),     # red
    STATUS_WARNING: (40, 200, 240), # amber
    STATUS_ALLOW: (90, 200, 90),    # green
    STATUS_IDLE: (170, 150, 90),    # calm blue-grey
    STATUS_SCANNING: (210, 170, 70),# cyan-ish: working / identifying
}

# Big banner Chinese labels (no emoji -> avoids tofu boxes on CJK fonts;
# a geometric icon is drawn separately).
_BANNER_TEXT = {
    STATUS_STOP: "紧急停止",
    STATUS_WARNING: "接近警告",
    STATUS_ALLOW: "允许接驳",
    STATUS_IDLE: "安全运行",
    STATUS_SCANNING: "识别中",
}

# Chinese explanations for common reasons (fallback: raw reason).
_REASON_CN = {
    "human_in_stop_zone": "检测到人体进入安全区",
    "limb_in_stop_zone": "检测到肢体进入安全区",
    "unknown_obstacle_in_stop_zone": "检测到未知障碍物",
    "human_in_warning_zone": "人员接近安全区",
    "object_in_warning_zone": "有物体接近安全区",
    "authorized_cart": "已授权接驳小车",
    "authorized_cart_in_waiting_window_and_safety_clear": "已授权接驳小车",
    "cart_not_verified:marker_not_visible": "未检测到授权标识",
    "docking_not_requested": "待机中",
    "": "",
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]

_font_cache: dict[int, object] = {}

@dataclass
class HudState:
    status: str                      # one of STATUS_*
    reason: str = ""                 # raw reason key
    fail_safe: bool = False          # watchdog fail-safe active
    blink_on: bool = True            # caller toggles for blinking STOP
    safety_zone_px: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1
    extra: str = ""                  # small free text (e.g. seq/age)

def decision_to_status(decision: Optional[str], motor: Optional[str], reason: Optional[str] = None) -> str:
    """Map a fusion decision / motor action / reason to a HUD status tier.

    Important: a WARNING decision has two very different meanings in fusion:
      - safety.warning (object_in_warning_zone) => real human/object proximity
        -> STATUS_WARNING (amber)
      - cart verification/stabilisation (cart_not_stable / cart_not_verified)
        => the system is still identifying/confirming the cart, NOT a danger
        -> STATUS_SCANNING (cyan)
    Mapping by decision enum alone wrongly showed 'cart confirming' as a human
    proximity warning (observed 2026-06-22 with a single static marker).
    """
    d = (decision or "").upper()
    m = (motor or "").upper()
    rl = (reason or "").lower()
    if d == "STOP" or m == "STOP":
        return STATUS_STOP
    # Cart identification / stabilisation in progress -> scanning, not danger.
    if "cart_not_stable" in rl or "cart_not_verified" in rl or "marker_not" in rl:
        return STATUS_SCANNING
    if "WARN" in d:
        # Genuine proximity warning only when it is about a zone object.
        if "warning_zone" in rl or rl == "":
            return STATUS_WARNING
        return STATUS_SCANNING
    if "ALLOW" in d or m == "ALLOW_CART_DOCKING":
        return STATUS_ALLOW
    return STATUS_IDLE

def _get_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    if _PIL_OK:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font

def reason_cn(reason: str) -> str:
    return _REASON_CN.get(reason, reason)

def render_hud(frame: np.ndarray, state: HudState) -> np.ndarray:
    """Composite the product HUD onto a BGR frame. Returns a new frame.

    Safe to call without PIL: falls back to no text (still draws zone/banner box)
    so it never crashes the live loop.
    """
    if frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    color = _COLORS.get(state.status, _COLORS[STATUS_IDLE])

    # --- dynamic safety zone highlight ---
    if state.safety_zone_px is not None:
        x0, y0, x1, y1 = state.safety_zone_px
        x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            import cv2
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            alpha = 0.28
            # brighter pulse when STOP is blinking on
            if state.status == STATUS_STOP and state.blink_on:
                alpha = 0.45
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)

    # --- top status banner ---
    banner_h = max(54, h // 9)
    import cv2
    # blinking: STOP banner dims when blink_off
    draw_banner = not (state.status == STATUS_STOP and not state.blink_on)
    banner_color = color if draw_banner else tuple(int(c * 0.45) for c in color)
    cv2.rectangle(out, (0, 0), (w, banner_h), banner_color, -1)

    if _PIL_OK:
        out = _draw_cjk(out, state, banner_h, w)
    out = _draw_status_icon(out, state, banner_h)
    return out

def _draw_status_icon(out: np.ndarray, state: HudState, banner_h: int) -> np.ndarray:
    """Draw a geometric status icon at the left of the banner (no emoji/tofu).

    STOP/fail-safe: octagon-ish filled circle; WARNING: triangle; ALLOW: check
    drawn as two strokes; IDLE: filled dot. All white on the coloured banner.
    """
    import cv2
    cy = banner_h // 2
    r = max(12, banner_h // 3)
    cx = 18 + r
    white = (255, 255, 255)
    status = STATUS_STOP if state.fail_safe else state.status
    if status == STATUS_STOP:
        cv2.circle(out, (cx, cy), r, white, 3)
        cv2.line(out, (cx - r // 2, cy - r // 2), (cx + r // 2, cy + r // 2), white, 3)
    elif status == STATUS_WARNING:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.polylines(out, [pts], True, white, 3)
        cv2.line(out, (cx, cy - r // 3), (cx, cy + r // 3), white, 3)
    elif status == STATUS_ALLOW:
        cv2.line(out, (cx - r, cy), (cx - r // 4, cy + r), white, 4)
        cv2.line(out, (cx - r // 4, cy + r), (cx + r, cy - r), white, 4)
    elif status == STATUS_SCANNING:
        # open ring + center dot: "identifying"
        cv2.circle(out, (cx, cy), r, white, 2)
        cv2.circle(out, (cx, cy), max(2, r // 4), white, -1)
    else:  # IDLE
        cv2.circle(out, (cx, cy), r // 2, white, -1)
    return out

def _draw_cjk(out_bgr: np.ndarray, state: HudState, banner_h: int, w: int) -> np.ndarray:
    img = Image.fromarray(out_bgr[:, :, ::-1])  # BGR->RGB
    draw = ImageDraw.Draw(img)

    banner_font = _get_font(max(28, banner_h - 18))
    sub_font = _get_font(max(18, banner_h // 3))

    text = _BANNER_TEXT.get(state.status, state.status)
    if state.fail_safe:
        text = "安全失联·已停机"
    # leave room on the left for the geometric status icon
    banner_size = banner_font.size if hasattr(banner_font, "size") else 28
    icon_w = max(12, banner_h // 3) * 2 + 28
    draw.text((icon_w, max(2, (banner_h - banner_size) // 2)),
              text, font=banner_font, fill=(255, 255, 255))

    # subtitle: Chinese reason + extra
    sub = reason_cn(state.reason)
    if state.fail_safe:
        sub = "未在限定时间内收到摄像头安全信号"
    if state.extra:
        sub = (sub + "   " + state.extra).strip()
    if sub:
        draw.text((20, banner_h + 8), sub, font=sub_font, fill=(235, 235, 235))

    return np.asarray(img)[:, :, ::-1].copy()  # RGB->BGR
