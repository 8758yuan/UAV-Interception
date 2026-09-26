"""Check target-class selection and the existing feature timing contract."""

import pytest

from ibvs_perception.yolo_detection import select_target_box


def test_highest_confidence_box_of_requested_class_is_selected() -> None:
    selected = select_target_box(
        [(1, 2, 11, 12), (20, 30, 40, 50), (4, 6, 14, 16)],
        [0.99, 0.71, 0.82],
        [0, 1, 1],
        target_class_id=1,
        image_width=100,
        image_height=100,
    )
    assert selected is not None
    assert selected.center == pytest.approx((9.0, 11.0))
    assert selected.area_px == pytest.approx(100.0)
    assert selected.confidence == pytest.approx(0.82)


def test_no_matching_box_has_no_fabricated_center() -> None:
    assert select_target_box(
        [(1, 2, 11, 12)], [0.95], [0],
        target_class_id=1, image_width=100, image_height=100,
    ) is None
    assert select_target_box(
        [], [], [], target_class_id=1, image_width=100, image_height=100,
    ) is None


def test_invalid_box_is_rejected_and_outside_box_is_clipped() -> None:
    selected = select_target_box(
        [(4, 5, 4, 12), (-5, 10, 105, 40)],
        [0.99, 0.80],
        [1, 1],
        target_class_id=1,
        image_width=100,
        image_height=100,
    )
    assert selected is not None
    assert (selected.x1, selected.x2) == (0.0, 100.0)
    assert selected.center == pytest.approx((50.0, 25.0))
