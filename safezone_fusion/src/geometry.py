from __future__ import annotations

from typing import Dict

from .models import Pose

def in_box(pose: Pose, box: Dict[str, float]) -> bool:
    return (
        box["x_min"] <= pose.x <= box["x_max"]
        and box["y_min"] <= pose.y <= box["y_max"]
        and box["z_min"] <= pose.z <= box["z_max"]
    )

def in_waiting_window(pose: Pose, window: Dict[str, float]) -> bool:
    return (
        window["x_min"] <= pose.x <= window["x_max"]
        and window["distance_min"] <= pose.z <= window["distance_max"]
        and abs(pose.yaw_deg) <= window["yaw_abs_max_deg"]
    )

def in_image_corridor(center_px, corridor: Dict[str, float],
                      frame_w: int, frame_h: int) -> bool:
    """Image-region docking corridor for a fixed overhead (top-down) camera.

    For a ceiling camera looking straight down at the cabinet face, "is the cart
    in the docking corridor" is naturally "is the marker inside a region of the
    image", which does not depend on fragile 3D distance/yaw estimation. The
    corridor is a normalized box (0..1) over the frame:

        {x_min, y_min, x_max, y_max}   # fractions of width/height

    Returns False when the marker center is unknown (cannot confirm -> not in
    corridor -> no exemption, which is the safe default).
    """
    if center_px is None or frame_w <= 0 or frame_h <= 0:
        return False
    cx, cy = center_px
    nx = cx / float(frame_w)
    ny = cy / float(frame_h)
    return (
        corridor["x_min"] <= nx <= corridor["x_max"]
        and corridor["y_min"] <= ny <= corridor["y_max"]
    )

def near_marker_region(obj_pose: Pose, marker_pose: Pose, region: Dict[str, float]) -> bool:
    return (
        abs(obj_pose.x - marker_pose.x) <= region["x_radius"]
        and abs(obj_pose.y - marker_pose.y) <= region["y_radius"]
        and abs(obj_pose.z - marker_pose.z) <= region["z_radius"]
    )
