"""Pure pinhole-camera and optical-frame geometry utilities."""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Rectified pinhole intrinsics for a right-down-forward optical frame."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        """Reject malformed calibration before it reaches the controller."""
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('camera width and height must be positive')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('camera intrinsics must be finite')
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError('camera focal lengths must be positive')


def intrinsics_from_horizontal_fov(
    width: int,
    height: int,
    horizontal_fov_rad: float,
) -> CameraIntrinsics:
    """Build centered square-pixel intrinsics from a Gazebo camera FOV."""
    if width <= 0 or height <= 0:
        raise ValueError('camera width and height must be positive')
    if not math.isfinite(horizontal_fov_rad) or not 0.0 < horizontal_fov_rad < math.pi:
        raise ValueError('horizontal_fov_rad must be in (0, pi)')
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )
    intrinsics.validate()
    return intrinsics


def normalized_pixel(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float]:
    """Convert a pixel centre to resolution-independent image coordinates."""
    intrinsics.validate()
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError('pixel coordinates must be finite')
    return (
        (u - intrinsics.cx) / intrinsics.fx,
        (v - intrinsics.cy) / intrinsics.fy,
    )


def low_pass_feature(
    previous_xy: Iterable[float],
    measured_xy: Iterable[float],
    elapsed_s: float,
    time_constant_s: float,
) -> Tuple[float, float]:
    """Filter normalized image coordinates without depending on frame rate."""
    previous = np.asarray(tuple(previous_xy), dtype=float)
    measured = np.asarray(tuple(measured_xy), dtype=float)
    if previous.shape != (2,) or measured.shape != (2,):
        raise ValueError('image coordinates must each contain two values')
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(measured)):
        raise ValueError('image coordinates must be finite')
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError('elapsed_s must be finite and nonnegative')
    if not math.isfinite(time_constant_s) or time_constant_s < 0.0:
        raise ValueError('time_constant_s must be finite and nonnegative')
    if time_constant_s == 0.0:
        return float(measured[0]), float(measured[1])
    alpha = -math.expm1(-elapsed_s / time_constant_s)
    filtered = previous + alpha * (measured - previous)
    return float(filtered[0]), float(filtered[1])


def project_optical_point(
    point_c: Iterable[float],
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float]:
    """Project a right-down-forward optical-frame point to pixel coordinates."""
    intrinsics.validate()
    x_c, y_c, z_c = _finite_vector(point_c, 'point_c')
    if z_c <= 0.0:
        raise ValueError('point_c must be in front of the camera')
    return (
        intrinsics.fx * x_c / z_c + intrinsics.cx,
        intrinsics.fy * y_c / z_c + intrinsics.cy,
    )


def optical_los_from_normalized(x_norm: float, y_norm: float) -> np.ndarray:
    """Return the unit target LOS in the right-down-forward optical frame."""
    if not math.isfinite(x_norm) or not math.isfinite(y_norm):
        raise ValueError('normalized image coordinates must be finite')
    vector = np.array((x_norm, y_norm, 1.0), dtype=float)
    return vector / np.linalg.norm(vector)


def default_camera_to_body_rotation() -> np.ndarray:
    """Map optical right-down-forward axes into PX4/ROS FLU body axes.

    Optical +x is body right (-y), optical +y is body down (-z), and optical
    +z is body forward (+x).  Columns are optical axes expressed in body.
    """
    return np.array(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        dtype=float,
    )


def camera_los_to_world(
    x_norm: float,
    y_norm: float,
    rotation_c_to_b: Iterable[float],
    rotation_b_to_e: Iterable[float],
) -> np.ndarray:
    """Map a detected normalized coordinate to an ENU unit LOS."""
    r_c_to_b = _rotation(rotation_c_to_b, 'rotation_c_to_b')
    r_b_to_e = _rotation(rotation_b_to_e, 'rotation_b_to_e')
    los_e = r_b_to_e @ r_c_to_b @ optical_los_from_normalized(x_norm, y_norm)
    return los_e / np.linalg.norm(los_e)


def red_hsv_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a red HSV mask without a cv2 runtime dependency.

    The Gazebo bridge can publish rgb8 or bgr8.  The caller normalizes either
    encoding to RGB before calling this vectorized implementation.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('rgb image must have shape (height, width, 3)')
    values = np.asarray(rgb, dtype=float) / 255.0
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-12
    red_max = nonzero & (maximum == values[:, :, 0])
    green_max = nonzero & (maximum == values[:, :, 1])
    blue_max = nonzero & (maximum == values[:, :, 2])
    hue[red_max] = np.mod(
        (values[:, :, 1][red_max] - values[:, :, 2][red_max])
        / delta[red_max],
        6.0,
    )
    hue[green_max] = (
        (values[:, :, 2][green_max] - values[:, :, 0][green_max])
        / delta[green_max]
    ) + 2.0
    hue[blue_max] = (
        (values[:, :, 0][blue_max] - values[:, :, 1][blue_max])
        / delta[blue_max]
    ) + 4.0
    hue /= 6.0
    saturation = np.divide(
        delta,
        maximum,
        out=np.zeros_like(delta),
        where=maximum > 1e-12,
    )
    return (((hue <= 0.05) | (hue >= 0.95)) & (saturation >= 0.45)
            & (maximum >= 0.25))


def _finite_vector(values: Iterable[float], name: str) -> Tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f'{name} must be a finite three-vector')
    return result


def _rotation(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (9,):
        array = array.reshape((3, 3))
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite 3x3 matrix')
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-6):
        raise ValueError(f'{name} must be orthonormal')
    if not math.isclose(float(np.linalg.det(array)), 1.0, abs_tol=1e-6):
        raise ValueError(f'{name} must have determinant +1')
    return array
