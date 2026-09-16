"""Coordinate-frame helpers shared by PX4 interface nodes."""

import math
from typing import Sequence, Tuple

import numpy as np


Vector3 = Tuple[float, float, float]


def ned_to_enu(vector: Sequence[float]) -> Vector3:
    """Convert an earth-frame vector from PX4 NED to ROS ENU."""
    north, east, down = _vector3(vector)
    return east, north, -down


def enu_to_ned(vector: Sequence[float]) -> Vector3:
    """Convert an earth-frame vector from ROS ENU to PX4 NED."""
    east, north, up = _vector3(vector)
    return north, east, -up


def frd_to_flu(vector: Sequence[float]) -> Vector3:
    """Convert a body-frame vector from PX4 FRD to ROS FLU."""
    forward, right, down = _vector3(vector)
    return forward, -right, -down


def flu_to_frd(vector: Sequence[float]) -> Vector3:
    """Convert a body-frame vector from ROS FLU to PX4 FRD."""
    forward, left, up = _vector3(vector)
    return forward, -left, -up


def tilt_from_body_to_ned_quaternion(q_wxyz: Sequence[float]) -> float:
    """Return body-down versus NED-down tilt for a Hamilton quaternion."""
    if len(q_wxyz) != 4:
        raise ValueError('quaternion must contain exactly four values')
    if not all(math.isfinite(float(value)) for value in q_wxyz):
        raise ValueError('quaternion values must be finite')

    w, x, y, z = (float(value) for value in q_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError('quaternion norm must be nonzero')
    w, x, y, z = (value / norm for value in (w, x, y, z))

    body_down_dot_ned_down = 1.0 - 2.0 * (x * x + y * y)
    clamped_dot = max(-1.0, min(1.0, body_down_dot_ned_down))
    return math.acos(clamped_dot)


def quaternion_wxyz_to_rotation(q_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a Hamilton body-to-world quaternion to a rotation matrix."""
    if len(q_wxyz) != 4:
        raise ValueError('quaternion must contain exactly four values')
    quaternion = np.asarray(q_wxyz, dtype=float)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError('quaternion values must be finite')
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError('quaternion norm must be nonzero')
    w, x, y, z = quaternion / norm
    return np.array((
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ),
        (
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ),
        (
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
    ))


def quaternion_wxyz_to_euler(q_wxyz: Sequence[float]) -> Vector3:
    """Return roll, pitch, yaw in radians for a body-to-world quaternion."""
    rotation = quaternion_wxyz_to_rotation(q_wxyz)
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def px4_quaternion_to_enu_flu_rotation(
    q_body_frd_to_ned_wxyz: Sequence[float],
) -> np.ndarray:
    """Convert PX4 body-FRD-to-NED attitude into body-FLU-to-ENU."""
    rotation_frd_to_ned = quaternion_wxyz_to_rotation(
        q_body_frd_to_ned_wxyz
    )
    ned_to_enu_matrix = np.array(((0, 1, 0), (1, 0, 0), (0, 0, -1)))
    flu_to_frd_matrix = np.diag((1, -1, -1))
    return ned_to_enu_matrix @ rotation_frd_to_ned @ flu_to_frd_matrix


def _vector3(vector: Sequence[float]) -> Vector3:
    if len(vector) != 3:
        raise ValueError('vector must contain exactly three values')
    result = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in result):
        raise ValueError('vector values must be finite')
    return result
