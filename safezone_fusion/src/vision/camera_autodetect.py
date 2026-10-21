from __future__ import annotations

"""Auto-detect cameras and assign RGB / NIR roles for foolproof startup.

On-site goal
------------
Operators should not have to remember ``--camera-index 0 --primary-role rgb
--secondary-index 1 --secondary-role nir``. This module scans the available
video devices, guesses each one's role (color vs mono/NIR) from the actual
frame, and produces a ready-to-use capture plan:

* 1 camera  -> single mode, role auto-detected
* 2 cameras -> dual mode, RGB as primary, NIR/mono as secondary
* 0 cameras -> explicit error (fail loud, not a silent black feed)

The role assignment is pure logic over probe results, so it is unit-testable
without any physical camera (inject fake probe results).
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CapturePlan:
    primary_index: int
    primary_role: str
    secondary_index: Optional[int] = None
    secondary_role: Optional[str] = None
    note: str = ""

    def to_cli_args(self) -> List[str]:
        args = ["--camera-index", str(self.primary_index), "--primary-role", self.primary_role]
        if self.secondary_index is not None:
            args += ["--secondary-index", str(self.secondary_index),
                     "--secondary-role", self.secondary_role or "nir"]
        return args

class NoCameraError(RuntimeError):
    """Raised when no usable camera is found -- must fail loud (safety)."""

def _normalize_role(role_guess: str) -> str:
    """Map probe role guesses to the two canonical roles used by the pipeline."""
    if role_guess == "rgb":
        return "rgb"
    if role_guess in ("mono_or_nir", "mono", "nir"):
        return "nir"
    return "unknown"

def plan_from_probes(probes) -> CapturePlan:
    """Build a CapturePlan from an iterable of CameraProbeResult-like objects.

    Rules (safety-biased, deterministic):
    * Only consider opened cameras.
    * Prefer an RGB camera as primary (marker/semantic detection lives there).
    * Pair a mono/NIR camera as secondary when present.
    * If two same-role cameras, keep lowest index primary, next secondary.
    * If roles are unknown, still run single mode on the lowest opened index
      (better to run with a best-effort feed than refuse), but note it.
    """
    opened = [p for p in probes if getattr(p, "opened", False)]
    if not opened:
        raise NoCameraError("no usable camera found (all probes failed to open)")

    opened = sorted(opened, key=lambda p: p.index)
    rgb = [p for p in opened if _normalize_role(p.role_guess) == "rgb"]
    nir = [p for p in opened if _normalize_role(p.role_guess) == "nir"]

    # Single camera
    if len(opened) == 1:
        p = opened[0]
        role = _normalize_role(p.role_guess)
        if role == "unknown":
            return CapturePlan(primary_index=p.index, primary_role="rgb",
                               note=f"single camera idx{p.index}, role unknown -> assume rgb")
        return CapturePlan(primary_index=p.index, primary_role=role,
                           note=f"single camera idx{p.index}, role={role}")

    # Two or more: prefer RGB primary + NIR secondary
    if rgb and nir:
        return CapturePlan(primary_index=rgb[0].index, primary_role="rgb",
                           secondary_index=nir[0].index, secondary_role="nir",
                           note=f"dual: rgb idx{rgb[0].index} + nir idx{nir[0].index}")
    # Two RGB-ish (or one rgb + unknowns): rgb primary, next opened secondary
    if rgb:
        secondary = next((p for p in opened if p.index != rgb[0].index), None)
        return CapturePlan(primary_index=rgb[0].index, primary_role="rgb",
                           secondary_index=secondary.index if secondary else None,
                           secondary_role=_normalize_role(secondary.role_guess) if secondary else None,
                           note="dual: rgb primary + best-effort secondary")
    # Only NIR/mono devices: run primary as nir, pair the second
    if len(nir) >= 2:
        return CapturePlan(primary_index=nir[0].index, primary_role="nir",
                           secondary_index=nir[1].index, secondary_role="nir",
                           note="dual: both mono/nir (no rgb detected)")
    if nir:
        return CapturePlan(primary_index=nir[0].index, primary_role="nir",
                           note="single usable: nir/mono only")

    # All unknown roles but multiple devices
    return CapturePlan(primary_index=opened[0].index, primary_role="rgb",
                       secondary_index=opened[1].index, secondary_role="nir",
                       note="roles unknown -> assume idx-order rgb+nir")

def autodetect_plan(max_index: int = 4, width: int | None = None,
                    height: int | None = None, snapshot_dir=None) -> CapturePlan:
    """Probe indices 0..max_index-1 with the real OpenCV probe and plan roles.

    Imported lazily so this module stays importable (and testable) on machines
    without OpenCV.
    """
    from .opencv_capture import probe_camera

    probes = []
    for i in range(max_index):
        try:
            r = probe_camera(i, width=width, height=height, snapshot_dir=snapshot_dir)
        except Exception:
            continue
        if getattr(r, "opened", False):
            probes.append(r)
    return plan_from_probes(probes)
