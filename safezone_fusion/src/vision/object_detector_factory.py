"""Factory for SafeZone object detectors.

The factory keeps demo/replay/regression code stable while allowing different
real detectors to be plugged in (mock, ROI motion prototype, future YOLO,
MediaPipe hands, depth-based detector, etc.). New detectors register by name
via :func:`register_object_detector` so we never need to edit the front-end
scripts again when adding a new one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from .mock_pipeline import MockObjectDetector
from .roi_motion_detector import RoiMotionObjectDetector
from .skin_motion_detector import SkinMotionObjectDetector

ObjectDetectorBuilder = Callable[[Dict[str, Any]], Any]

_REGISTRY: Dict[str, ObjectDetectorBuilder] = {}

def register_object_detector(name: str, builder: ObjectDetectorBuilder) -> None:
    """Register a builder for an object detector backend.

    Builders receive the raw config dict (typically a sub-section of
    ``config.yaml``) and must return an object that exposes
    ``detect(packet) -> list[ObjectObservation]``.
    """

    if not name:
        raise ValueError("object detector name must be non-empty")
    _REGISTRY[name] = builder

def available_object_detectors() -> Iterable[str]:
    return tuple(_REGISTRY.keys())

def _build_mock(_: Dict[str, Any]) -> MockObjectDetector:
    return MockObjectDetector([[]] * 1000000)

def _build_roi_motion(cfg: Dict[str, Any]) -> RoiMotionObjectDetector:
    cfg = dict(cfg or {})
    cfg.setdefault("enabled", True)
    return RoiMotionObjectDetector.from_dict(cfg)

def _build_skin_motion(cfg: Dict[str, Any]) -> SkinMotionObjectDetector:
    cfg = dict(cfg or {})
    cfg.setdefault("enabled", True)
    return SkinMotionObjectDetector.from_dict(cfg)

def _build_mediapipe_hand(cfg: Dict[str, Any]):  # pragma: no cover - import-guarded
    """Optional MediaPipe Hands backend.

    This backend is intentionally optional because it requires the dedicated
    ``safezone`` conda environment. Selecting it explicitly should enable it
    immediately, just like ``roi_motion`` and ``skin_motion``.
    """

    cfg = dict(cfg or {})
    cfg.setdefault("enabled", True)
    try:
        from .mediapipe_hand_detector import MediaPipeHandObjectDetector  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mediapipe_hand backend is not available in this environment: "
            f"{exc}. Install a compatible mediapipe build before selecting it."
        ) from exc
    return MediaPipeHandObjectDetector.from_dict(cfg)

register_object_detector("mock", _build_mock)
register_object_detector("none", _build_mock)
register_object_detector("roi_motion", _build_roi_motion)
register_object_detector("skin_motion", _build_skin_motion)
register_object_detector("mediapipe_hand", _build_mediapipe_hand)

def build_object_detector(name: str, cfg: Dict[str, Any] | None = None) -> Any:
    """Build an object detector by registry name.

    Unknown names fall back to the mock detector and a warning is left in the
    caller's output via ``ValueError`` is too aggressive here; we keep the
    pipeline alive but mark the choice clearly.
    """

    name = (name or "mock").strip().lower()
    builder = _REGISTRY.get(name)
    if builder is None:
        raise KeyError(f"unknown object detector: {name!r}; available={sorted(_REGISTRY)}")
    return builder(cfg or {})

def resolve_object_detector(config: Dict[str, Any], roi_motion_force_enabled: bool = False) -> tuple[str, Any]:
    """Pick the right detector for the active ``config`` dict.

    Priority:

    1. ``roi_motion_force_enabled`` (CLI/regression flag) wins, so explicit
       intent to test the ROI motion prototype is never silently overridden
       by a saved ``object_detector.name``.
    2. ``config['object_detector']['name']`` if set.
    3. ``roi_motion`` section if explicitly enabled in YAML.
    4. ``mock`` as the safe default.
    """

    object_detector_cfg = config.get("object_detector") or {}
    roi_motion_cfg = config.get("roi_motion") or {}

    if roi_motion_force_enabled:
        return "roi_motion", build_object_detector("roi_motion", roi_motion_cfg)

    name = (object_detector_cfg.get("name") or "").strip().lower()
    if name:
        backend_cfg = object_detector_cfg.get("config")
        if backend_cfg is None:
            if name == "roi_motion":
                backend_cfg = roi_motion_cfg
            elif name == "skin_motion":
                backend_cfg = config.get("skin_motion") or {}
            else:
                backend_cfg = {}
        return name, build_object_detector(name, backend_cfg)

    if roi_motion_cfg.get("enabled"):
        return "roi_motion", build_object_detector("roi_motion", roi_motion_cfg)

    return "mock", build_object_detector("mock", {})

def _backend_cfg_for(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the config sub-section a named backend should be built from."""
    name = (name or "").strip().lower()
    if name == "roi_motion":
        return config.get("roi_motion") or {}
    if name == "skin_motion":
        return config.get("skin_motion") or {}
    if name == "mediapipe_hand":
        return config.get("mediapipe_hand") or {}
    return {}

def resolve_object_detectors(
    config: Dict[str, Any], names: Iterable[str]
) -> tuple[list[str], list[Any]]:
    """Build a *list* of object detectors from explicit backend names.

    Enables layered detection, e.g. ``["mediapipe_hand", "roi_motion"]`` so a
    semantic hand detector and a motion safety-net run together. Each detector
    is built with ``enabled=True`` forced, since selecting it by name is itself
    the intent to enable it (mirrors the single-backend factory behaviour).
    Returns (resolved_names, detectors) preserving order.
    """
    resolved_names: list[str] = []
    detectors: list[Any] = []
    for raw in names:
        name = (raw or "").strip().lower()
        if not name:
            continue
        cfg = dict(_backend_cfg_for(name, config))
        # Selecting a backend by name IS the intent to enable it, so force enable.
        # NOTE: must be assignment, not setdefault -- config blocks often ship
        # enabled:false, and setdefault would leave that False in place,
        # silently disabling the detector (observed 2026-06-22).
        cfg["enabled"] = True
        resolved_names.append(name)
        detectors.append(build_object_detector(name, cfg))
    return resolved_names, detectors
