"""Image-feature IBVS law with no target position or velocity input."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ibvs_control.so3_controller import (
    attitude_rate_feedback,
    rotation_between,
)


@dataclass(frozen=True)
class VisualIbvsConfig:
    """Tuning for the target-state-free visual interception outer loop."""

    mass_kg: float
    thrust_max_n: float
    k_b: float
    los_rate_gain: float
    image_center_rate_gain: float
    attitude_rate_gain: float
    speed_gain: float
    cruise_speed_m_s: float
    terminal_speed_m_s: float
    max_approach_acceleration_m_s2: float
    max_approach_deceleration_m_s2: float
    transverse_velocity_gain: float
    max_transverse_acceleration_m_s2: float
    vertical_velocity_gain: float
    max_vertical_correction_m_s2: float
    terminal_area_ratio: float
    full_speed_image_error: float
    stop_approach_image_error: float
    max_command_tilt_rad: float
    gravity_m_s2: float = 9.80665

    def validate(self) -> None:
        """Reject gains and limits that cannot form a safe command."""
        positive = {
            'mass_kg': self.mass_kg,
            'thrust_max_n': self.thrust_max_n,
            'k_b': self.k_b,
            'los_rate_gain': self.los_rate_gain,
            'image_center_rate_gain': self.image_center_rate_gain,
            'attitude_rate_gain': self.attitude_rate_gain,
            'speed_gain': self.speed_gain,
            'cruise_speed_m_s': self.cruise_speed_m_s,
            'terminal_speed_m_s': self.terminal_speed_m_s,
            'max_approach_acceleration_m_s2': (
                self.max_approach_acceleration_m_s2
            ),
            'max_approach_deceleration_m_s2': (
                self.max_approach_deceleration_m_s2
            ),
            'transverse_velocity_gain': self.transverse_velocity_gain,
            'max_transverse_acceleration_m_s2': (
                self.max_transverse_acceleration_m_s2
            ),
            'vertical_velocity_gain': self.vertical_velocity_gain,
            'max_vertical_correction_m_s2': (
                self.max_vertical_correction_m_s2
            ),
            'terminal_area_ratio': self.terminal_area_ratio,
            'full_speed_image_error': self.full_speed_image_error,
            'stop_approach_image_error': self.stop_approach_image_error,
            'max_command_tilt_rad': self.max_command_tilt_rad,
            'gravity_m_s2': self.gravity_m_s2,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0.0 < self.k_b < 2.0:
            raise ValueError('k_b must be within (0, 2)')
        if not 0.0 < self.terminal_area_ratio < 1.0:
            raise ValueError('terminal_area_ratio must be within (0, 1)')
        if self.stop_approach_image_error <= self.full_speed_image_error:
            raise ValueError(
                'stop_approach_image_error must exceed '
                'full_speed_image_error'
            )
        if self.max_command_tilt_rad >= math.pi / 2.0:
            raise ValueError('max_command_tilt_rad must be less than pi/2')


@dataclass(frozen=True)
class VisualIbvsResult:
    """One command computed from pixels and interceptor onboard state."""

    los_b: np.ndarray
    los_e: np.ndarray
    z1: float
    barrier_margin: float
    area_ratio: float
    image_error: float
    alignment_factor: float
    approach_speed_m_s: float
    desired_approach_speed_m_s: float
    approach_acceleration_m_s2: float
    transverse_velocity_e: np.ndarray
    transverse_correction_e: np.ndarray
    vertical_correction_m_s2: float
    acceleration_d_e: np.ndarray
    attitude_d_b_to_e: np.ndarray
    omega_los_b: np.ndarray
    omega_attitude_b: np.ndarray
    omega_d_b: np.ndarray
    thrust_n: float
    thrust_saturated: bool


def compute_visual_ibvs(
    x_norm: float,
    y_norm: float,
    area_ratio: float,
    attitude_b_to_e: Sequence[Sequence[float]],
    interceptor_velocity_e: Sequence[float],
    config: VisualIbvsConfig,
) -> VisualIbvsResult:
    """
    Compute the paper-style LOS/attitude command from image features.

    The only target-dependent inputs are normalized image coordinates and
    segmented image area.  Interceptor attitude and velocity are onboard PX4
    estimates; no target position, range, velocity, or world pose is used.
    """
    config.validate()
    values = (x_norm, y_norm, area_ratio)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('visual features must be finite')
    if area_ratio <= 0.0 or area_ratio > 1.0:
        raise ValueError('area_ratio must be within (0, 1]')
    rotation = _rotation3(attitude_b_to_e)
    velocity_e = _vector3(interceptor_velocity_e, 'interceptor_velocity_e')

    # Optical right/down/forward -> body FLU forward/left/up.  This is the
    # target unit vector n_t in paper equation (4), obtained only from pixels.
    los_b = _unit3((1.0, -x_norm, -y_norm), 'image LOS')
    los_e = rotation @ los_b
    designed_los_e = rotation[:, 0]
    z1 = 1.0 - float(designed_los_e @ los_e)
    denominator = config.k_b**2 - z1**2
    if z1 < 0.0 or denominator <= 1e-9:
        raise ValueError('image LOS is outside the barrier domain')
    barrier_gain = config.los_rate_gain * z1 / denominator
    image_cross = rotation.T @ np.cross(designed_los_e, los_e)
    omega_los_b = (
        barrier_gain + config.image_center_rate_gain
    ) * image_cross

    closeness = min(area_ratio / config.terminal_area_ratio, 1.0)
    image_error = math.hypot(x_norm, y_norm)
    alignment_factor = float(np.clip(
        (
            config.stop_approach_image_error - image_error
        ) / (
            config.stop_approach_image_error
            - config.full_speed_image_error
        ),
        0.0,
        1.0,
    ))
    scheduled_speed = (
        config.cruise_speed_m_s
        + closeness
        * (config.terminal_speed_m_s - config.cruise_speed_m_s)
    )
    desired_speed = alignment_factor * scheduled_speed
    approach_speed = float(velocity_e @ los_e)
    raw_acceleration = config.speed_gain * (desired_speed - approach_speed)
    approach_acceleration = float(np.clip(
        raw_acceleration,
        -config.max_approach_deceleration_m_s2,
        config.max_approach_acceleration_m_s2,
    ))

    # Driving only the scalar speed along LOS leaves any pre-existing
    # cross-track velocity untouched.  That becomes a large image error near
    # the target even when the target was centred for most of the approach.
    # Dampen the interceptor's own velocity perpendicular to the camera LOS;
    # this uses neither target velocity nor reconstructed range/position.
    transverse_velocity_e = velocity_e - approach_speed * los_e
    transverse_correction_e = (
        -config.transverse_velocity_gain * transverse_velocity_e
    )
    transverse_norm = float(np.linalg.norm(transverse_correction_e))
    if transverse_norm > config.max_transverse_acceleration_m_s2:
        transverse_correction_e *= (
            config.max_transverse_acceleration_m_s2 / transverse_norm
        )

    # A bounded damping term from the interceptor's own vertical velocity
    # keeps attitude transients from accumulating altitude loss.
    vertical_correction = float(np.clip(
        -config.vertical_velocity_gain * velocity_e[2],
        -config.max_vertical_correction_m_s2,
        config.max_vertical_correction_m_s2,
    ))
    acceleration_d_e = (
        approach_acceleration * los_e
        + transverse_correction_e
        + np.array((0.0, 0.0, vertical_correction))
    )
    gravity_e = np.array((0.0, 0.0, -config.gravity_m_s2))
    desired_force_e = config.mass_kg * (acceleration_d_e - gravity_e)
    thrust_direction_d_e = _limit_tilt_direction(
        _unit3(desired_force_e, 'desired force'),
        config.max_command_tilt_rad,
    )
    current_thrust_direction_e = rotation[:, 2]
    attitude_d = (
        rotation_between(
            current_thrust_direction_e,
            thrust_direction_d_e,
        )
        @ rotation
    )
    omega_attitude_b = config.attitude_rate_gain * attitude_rate_feedback(
        attitude_d,
        rotation,
    )
    omega_d_b = omega_los_b + omega_attitude_b

    raw_thrust_n = float(current_thrust_direction_e @ desired_force_e)
    thrust_n = min(max(raw_thrust_n, 0.0), config.thrust_max_n)
    return VisualIbvsResult(
        los_b=los_b,
        los_e=los_e,
        z1=z1,
        barrier_margin=config.k_b - abs(z1),
        area_ratio=area_ratio,
        image_error=image_error,
        alignment_factor=alignment_factor,
        approach_speed_m_s=approach_speed,
        desired_approach_speed_m_s=desired_speed,
        approach_acceleration_m_s2=approach_acceleration,
        transverse_velocity_e=transverse_velocity_e,
        transverse_correction_e=transverse_correction_e,
        vertical_correction_m_s2=vertical_correction,
        acceleration_d_e=acceleration_d_e,
        attitude_d_b_to_e=attitude_d,
        omega_los_b=omega_los_b,
        omega_attitude_b=omega_attitude_b,
        omega_d_b=omega_d_b,
        thrust_n=thrust_n,
        thrust_saturated=not math.isclose(thrust_n, raw_thrust_n),
    )


def terminal_visual_ready(
    area_ratio: float,
    x_norm: float,
    y_norm: float,
    minimum_area_ratio: float,
    maximum_center_error: float,
) -> bool:
    """Decide terminal entry strictly from target scale and image error."""
    values = (
        area_ratio,
        x_norm,
        y_norm,
        minimum_area_ratio,
        maximum_center_error,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return (
        area_ratio >= minimum_area_ratio
        and math.hypot(x_norm, y_norm) <= maximum_center_error
    )


def recent_close_target_lost(
    area_ratio: float,
    x_norm: float,
    y_norm: float,
    observation_age_s: float,
    minimum_area_ratio: float,
    maximum_center_error: float,
    maximum_age_s: float,
) -> bool:
    """Recognize a close, centered target disappearing from the live image."""
    values = (
        area_ratio,
        x_norm,
        y_norm,
        observation_age_s,
        minimum_area_ratio,
        maximum_center_error,
        maximum_age_s,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return (
        0.0 <= observation_age_s <= maximum_age_s
        and area_ratio >= minimum_area_ratio
        and math.hypot(x_norm, y_norm) <= maximum_center_error
    )


def _limit_tilt_direction(direction: np.ndarray, max_tilt_rad: float) -> np.ndarray:
    unit = _unit3(direction, 'thrust direction')
    minimum_vertical = math.cos(max_tilt_rad)
    if float(unit[2]) >= minimum_vertical:
        return unit
    horizontal = np.array((unit[0], unit[1], 0.0))
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-12:
        return np.array((0.0, 0.0, 1.0))
    return np.array((
        math.sin(max_tilt_rad) * horizontal[0] / horizontal_norm,
        math.sin(max_tilt_rad) * horizontal[1] / horizontal_norm,
        minimum_vertical,
    ))


def _vector3(values: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite three-vector')
    return value


def _unit3(values: Sequence[float], name: str) -> np.ndarray:
    value = _vector3(values, name)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f'{name} must be nonzero')
    return value / norm


def _rotation3(values: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError('attitude must be a finite 3x3 matrix')
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-7):
        raise ValueError('attitude must be orthonormal')
    if not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=1e-7):
        raise ValueError('attitude must have determinant +1')
    return value
