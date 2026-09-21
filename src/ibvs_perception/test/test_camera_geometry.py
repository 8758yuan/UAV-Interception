"""Regression tests for P3/P4 camera geometry independent of ROS runtime."""

import numpy as np
import pytest

from ibvs_perception.camera_geometry import (
    camera_los_to_world,
    default_camera_to_body_rotation,
    intrinsics_from_horizontal_fov,
    low_pass_feature,
    normalized_pixel,
    project_optical_point,
    red_hsv_mask,
)
from ibvs_perception.red_target_detector import _delayed_release_ns


def test_optical_projection_has_expected_pixel_directions() -> None:
    """Optical +x is image right and optical +y is image down."""
    intrinsics = intrinsics_from_horizontal_fov(640, 480, 2.0 * np.pi / 3.0)
    centre = project_optical_point((0.0, 0.0, 2.0), intrinsics)
    right = project_optical_point((0.2, 0.0, 2.0), intrinsics)
    down = project_optical_point((0.0, 0.2, 2.0), intrinsics)
    assert centre == pytest.approx((intrinsics.cx, intrinsics.cy))
    assert right[0] > centre[0]
    assert down[1] > centre[1]
    assert normalized_pixel(*centre, intrinsics) == pytest.approx((0.0, 0.0))


def test_default_camera_rotation_maps_centre_to_body_forward() -> None:
    """A centred front-camera detection gives an ENU/body-forward LOS."""
    los = camera_los_to_world(
        0.0,
        0.0,
        default_camera_to_body_rotation(),
        np.eye(3),
    )
    assert los == pytest.approx((1.0, 0.0, 0.0))


def test_feature_low_pass_rejects_frame_rate_dependent_tuning() -> None:
    once = low_pass_feature((0.0, 0.0), (1.0, -1.0), 0.1, 0.08)
    twice = low_pass_feature((0.0, 0.0), (1.0, -1.0), 0.05, 0.08)
    twice = low_pass_feature(twice, (1.0, -1.0), 0.05, 0.08)
    assert twice == pytest.approx(once)
    assert 0.0 < once[0] < 1.0
    assert -1.0 < once[1] < 0.0


def test_red_hsv_mask_rejects_nonred_pixels() -> None:
    """Target segmentation accepts both ends of red hue and excludes green."""
    image = np.array(
        [[[255, 0, 0], [255, 20, 35]], [[0, 255, 0], [30, 30, 30]]],
        dtype=np.uint8,
    )
    assert red_hsv_mask(image).tolist() == [[True, True], [False, False]]


def test_feature_release_enforces_capture_to_publish_delay() -> None:
    assert _delayed_release_ns(1_000_000_000, 1_010_000_000, 0.08) == (
        1_080_000_000
    )
    # If processing itself already exceeded 80 ms, do not add a second delay.
    assert _delayed_release_ns(1_000_000_000, 1_090_000_000, 0.08) == (
        1_090_000_000
    )
