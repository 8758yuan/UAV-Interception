"""Small, ROS-independent helpers for selecting one YOLO target box."""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class TargetBox:
    """A selected detection in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area_px(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def select_target_box(
    xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    *,
    target_class_id: int,
    image_width: int,
    image_height: int,
) -> TargetBox | None:
    """Select the highest-confidence box of the configured target class."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image dimensions must be positive')
    if not (len(xyxy) == len(confidences) == len(class_ids)):
        raise ValueError('YOLO box fields must have matching lengths')

    candidates = []
    for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
        if int(class_id) != target_class_id:
            continue
        values = tuple(float(value) for value in coordinates)
        confidence = float(confidence)
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            continue
        if (
            not math.isfinite(confidence)
            or confidence < 0.0
            or confidence > 1.0
        ):
            continue
        x1 = min(max(values[0], 0.0), float(image_width))
        y1 = min(max(values[1], 0.0), float(image_height))
        x2 = min(max(values[2], 0.0), float(image_width))
        y2 = min(max(values[3], 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            TargetBox(x1, y1, x2, y2, confidence, target_class_id)
        )

    if not candidates:
        return None
    # Tie breaks keep selection deterministic across equivalent results.
    return max(
        candidates,
        key=lambda box: (box.confidence, box.area_px, -box.x1, -box.y1),
    )


def delayed_release_ns(
    capture_ns: int, arrival_ns: int, delay_s: float
) -> int:
    """Retain the existing capture-to-publication delay contract."""
    if not math.isfinite(delay_s) or delay_s < 0.0:
        raise ValueError('delay_s must be finite and nonnegative')
    delay_ns = int(round(delay_s * 1e9))
    if capture_ns <= 0:
        return arrival_ns + delay_ns
    return max(arrival_ns, capture_ns + delay_ns)
