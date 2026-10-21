from __future__ import annotations

from typing import Any, Dict, Optional

from .geometry import in_box
from .models import CartAuthResult, MarkerObservation

class CartAuthenticator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authorized_carts = config.get("authorized_carts", {})

    def find_cart_by_marker(self, marker_id: int) -> Optional[tuple[str, Dict[str, Any]]]:
        for cart_id, spec in self.authorized_carts.items():
            if spec.get("marker_id") == marker_id:
                return cart_id, spec
        return None

    def verify(self, marker: Optional[MarkerObservation]) -> CartAuthResult:
        if marker is None or not marker.visible:
            return CartAuthResult(False, reason="marker_not_visible")

        found = self.find_cart_by_marker(marker.marker_id)
        if not found:
            return CartAuthResult(False, marker_id=marker.marker_id, reason="marker_not_authorized")

        cart_id, spec = found
        if not spec.get("enabled", False):
            return CartAuthResult(False, cart_id=cart_id, marker_id=marker.marker_id, reason="cart_disabled")

        required_frames = int(spec.get("required_stable_frames", 5))
        if marker.stable_frames < required_frames:
            return CartAuthResult(
                False,
                cart_id=cart_id,
                marker_id=marker.marker_id,
                reason=f"marker_not_stable:{marker.stable_frames}/{required_frames}",
                pose=marker.pose,
            )

        area = spec.get("allowed_marker_area")
        if area and not in_box(marker.pose, area):
            return CartAuthResult(
                False,
                cart_id=cart_id,
                marker_id=marker.marker_id,
                reason="marker_out_of_allowed_area",
                pose=marker.pose,
            )

        return CartAuthResult(
            True,
            cart_id=cart_id,
            marker_id=marker.marker_id,
            reason="authorized_marker_verified",
            pose=marker.pose,
        )
