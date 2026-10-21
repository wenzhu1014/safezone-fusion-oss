from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

StereoLayout = Literal["single", "side_by_side", "top_bottom"]

@dataclass
class StereoSplitResult:
    layout: StereoLayout
    left_rgb: Any
    right_rgb: Any | None
    reason: str

class StereoFrameSplitter:
    """Split a camera frame into left/right views when a stereo module exposes one combined UVC stream.

    Supported layouts:
    - single: one ordinary view; right_rgb=None
    - side_by_side: left/right views horizontally concatenated
    - top_bottom: left/right views vertically concatenated
    """

    def __init__(self, layout: StereoLayout = "single"):
        self.layout = layout

    def split(self, frame: Any) -> StereoSplitResult:
        if frame is None:
            return StereoSplitResult("single", None, None, "empty_frame")
        h, w = frame.shape[:2]
        if self.layout == "side_by_side":
            mid = w // 2
            return StereoSplitResult("side_by_side", frame[:, :mid].copy(), frame[:, mid:].copy(), f"split width {w}->{mid}+{w-mid}")
        if self.layout == "top_bottom":
            mid = h // 2
            return StereoSplitResult("top_bottom", frame[:mid, :].copy(), frame[mid:, :].copy(), f"split height {h}->{mid}+{h-mid}")
        return StereoSplitResult("single", frame, None, f"single frame {w}x{h}")
