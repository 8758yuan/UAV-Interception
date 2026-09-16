import math

import numpy as np
import pytest

from ibvs_control.frames import (
    enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    px4_quaternion_to_enu_flu_rotation,
    quaternion_wxyz_to_euler,
    quaternion_wxyz_to_rotation,
    tilt_from_body_to_ned_quaternion,
)


def test_ned_enu_axis_mapping_and_round_trip() -> None:
    """North/east swap and down/up sign follow REP-103 conventions."""
    ned = (1.0, 2.0, 3.0)
    assert ned_to_enu(ned) == (2.0, 1.0, -3.0)
    assert enu_to_ned(ned_to_enu(ned)) == ned


def test_frd_flu_axis_mapping_and_round_trip() -> None:
    """Forward is shared while right/down become left/up."""
    frd = (1.0, 2.0, 3.0)
    assert frd_to_flu(frd) == (1.0, -2.0, -3.0)
    assert flu_to_frd(frd_to_flu(frd)) == frd


def test_level_quaternion_has_zero_tilt() -> None:
    """The identity body-to-NED attitude is level."""
    assert tilt_from_body_to_ned_quaternion((1.0, 0.0, 0.0, 0.0)) == 0.0


def test_yaw_does_not_change_tilt() -> None:
    """Pure yaw leaves the body-down axis aligned with NED down."""
    half_angle = math.pi / 4.0
    quaternion = (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    assert tilt_from_body_to_ned_quaternion(quaternion) == 0.0


def test_roll_angle_is_reported_as_tilt() -> None:
    """A pure 30-degree roll has a 30-degree combined tilt."""
    half_angle = math.radians(15.0)
    quaternion = (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0)
    tilt = tilt_from_body_to_ned_quaternion(quaternion)
    assert tilt == pytest.approx(math.radians(30.0))


def test_invalid_frame_inputs_are_rejected() -> None:
    """Malformed or non-finite vectors must not enter flight calculations."""
    with pytest.raises(ValueError):
        ned_to_enu((1.0, 2.0))
    with pytest.raises(ValueError):
        tilt_from_body_to_ned_quaternion((0.0, 0.0, 0.0, 0.0))


def test_quaternion_rotation_preserves_axes_and_so3() -> None:
    """A 90-degree yaw rotates body X onto world Y."""
    half_angle = math.pi / 4.0
    rotation = quaternion_wxyz_to_rotation(
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    )
    assert rotation @ np.array((1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_quaternion_euler_reports_roll_pitch_yaw() -> None:
    half_angle = math.pi / 4.0
    euler = quaternion_wxyz_to_euler(
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    )
    assert euler == pytest.approx((0.0, 0.0, math.pi / 2.0))


def test_px4_yaw_zero_points_flu_forward_toward_enu_north() -> None:
    """PX4 NED north maps to ENU +Y while body FLU stays right-handed."""
    rotation = px4_quaternion_to_enu_flu_rotation((1.0, 0.0, 0.0, 0.0))
    assert rotation @ np.array((1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert rotation @ np.array((0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)
