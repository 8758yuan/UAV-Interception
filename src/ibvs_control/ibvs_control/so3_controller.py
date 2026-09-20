"""Pure SO(3) interception controller equations from the paper."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class OuterLoopConfig:
    """Tuning and vehicle parameters used by the 50 Hz outer loop."""

    k1: float
    k2: float
    k_b: float
    mass_kg: float
    thrust_max_n: float
    gravity_m_s2: float = 9.80665
    max_command_tilt_rad: float = math.radians(20.0)
    # ``None`` keeps the exact paper equations for offline analysis.  A real
    # airframe can opt into these envelopes because the paper assumes an
    # effectively unlimited thrust/acceleration budget.
    max_acceleration_m_s2: Optional[float] = None
    max_vertical_acceleration_m_s2: Optional[float] = None

    def validate(self) -> None:
        """Reject nonphysical gains and limits."""
        positive = (self.k1, self.k2, self.k_b, self.mass_kg)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('k1, k2, k_b, and mass_kg must be positive')
        if not 0.0 < self.k_b < 2.0:
            raise ValueError('k_b must be within (0, 2)')
        if not math.isfinite(self.thrust_max_n) or self.thrust_max_n <= 0.0:
            raise ValueError('thrust_max_n must be finite and positive')
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError('gravity_m_s2 must be finite and positive')
        if (
            not math.isfinite(self.max_command_tilt_rad)
            or self.max_command_tilt_rad <= 0.0
            or self.max_command_tilt_rad >= math.pi / 2.0
        ):
            raise ValueError(
                'max_command_tilt_rad must be within (0, pi/2)'
            )
        for name, value in (
            ('max_acceleration_m_s2', self.max_acceleration_m_s2),
            (
                'max_vertical_acceleration_m_s2',
                self.max_vertical_acceleration_m_s2,
            ),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class OuterLoopResult:
    """Outputs frozen by the 50 Hz collinear outer loop."""

    z1: float
    z2_e: np.ndarray
    acceleration_d_e: np.ndarray
    thrust_direction_d_e: np.ndarray
    attitude_d_b_to_e: np.ndarray
    omega1_b: np.ndarray
    thrust_n: float
    thrust_saturated: bool


@dataclass(frozen=True)
class InnerLoopResult:
    """Paper (23), (26), and (28) evaluated at the current attitude."""

    omega2_b: np.ndarray
    omega_d_b: np.ndarray
    thrust_n: float
    thrust_saturated: bool
    rate_saturated: bool


def compute_outer_loop(
    p_r_e: Sequence[float],
    v_r_e: Sequence[float],
    los_e: Sequence[float],
    designed_los_e: Sequence[float],
    attitude_b_to_e: Sequence[Sequence[float]],
    config: OuterLoopConfig,
    *,
    target_acceleration_e: Sequence[float] = (0.0, 0.0, 0.0),
    drag_force_e: Sequence[float] = (0.0, 0.0, 0.0),
) -> OuterLoopResult:
    """Evaluate paper equations (13), (19), and (21)--(23)."""
    config.validate()
    p_r = _vector3(p_r_e, 'p_r_e')
    v_r = _vector3(v_r_e, 'v_r_e')
    n_t = _unit3(los_e, 'los_e')
    n_td = _unit3(designed_los_e, 'designed_los_e')
    rotation = _rotation3(attitude_b_to_e, 'attitude_b_to_e')
    a_target = _vector3(target_acceleration_e, 'target_acceleration_e')
    drag = _vector3(drag_force_e, 'drag_force_e')

    distance_m = float(np.linalg.norm(p_r))
    if distance_m <= 1e-6:
        raise ValueError('relative distance is too small for the controller')
    z1 = 1.0 - float(n_td @ n_t)
    denominator = config.k_b**2 - z1**2
    if z1 < 0.0 or denominator <= 1e-9:
        raise ValueError('LOS error is outside the barrier domain')
    barrier_gain = z1 / denominator

    z2 = v_r + config.k1 * p_r
    projection = -np.eye(3) + np.outer(n_t, n_t)
    acceleration_d = (
        -config.k1 * v_r
        - config.k2 * z2
        - p_r
        + barrier_gain * config.mass_kg / distance_m * projection @ n_td
        + a_target
    )
    acceleration_d = _limit_acceleration(
        acceleration_d,
        config.max_acceleration_m_s2,
        config.max_vertical_acceleration_m_s2,
    )
    gravity = np.array((0.0, 0.0, -config.gravity_m_s2))
    desired_force = config.mass_kg * (acceleration_d - gravity) - drag
    n_fd = _limit_tilt_direction(
        _unit3(desired_force, 'desired force'),
        config.max_command_tilt_rad,
    )

    n_f = rotation[:, 2]
    attitude_d = rotation_between(n_f, n_fd) @ rotation
    raw_thrust_n = float(n_f @ desired_force)
    thrust_n = min(max(raw_thrust_n, 0.0), config.thrust_max_n)
    omega1_b = barrier_gain * rotation.T @ np.cross(n_td, n_t)
    return OuterLoopResult(
        z1=z1,
        z2_e=z2,
        acceleration_d_e=acceleration_d,
        thrust_direction_d_e=n_fd,
        attitude_d_b_to_e=attitude_d,
        omega1_b=omega1_b,
        thrust_n=thrust_n,
        thrust_saturated=not math.isclose(thrust_n, raw_thrust_n),
    )


def compute_inner_loop(
    outer: OuterLoopResult,
    attitude_b_to_e: Sequence[Sequence[float]],
    config: OuterLoopConfig,
    omega_max_rad_s: float,
    *,
    drag_force_e: Sequence[float] = (0.0, 0.0, 0.0),
    minimum_thrust_n: float = 0.0,
) -> InnerLoopResult:
    """
    Evaluate the 200 Hz part of Algorithm 1 using paper equations.

    The 50 Hz outer-loop values ``a_d``, ``R_d`` and ``omega_1`` are held
    between updates.  The current attitude is used here for equations (23)
    and (26), before the norm saturation in equations (28)-(29).
    """
    config.validate()
    if (
        not math.isfinite(minimum_thrust_n)
        or minimum_thrust_n < 0.0
        or minimum_thrust_n > config.thrust_max_n
    ):
        raise ValueError(
            'minimum_thrust_n must be within [0, thrust_max_n]'
        )
    rotation = _rotation3(attitude_b_to_e, 'attitude_b_to_e')
    drag = _vector3(drag_force_e, 'drag_force_e')
    omega2_b = attitude_rate_feedback(
        outer.attitude_d_b_to_e,
        rotation,
    )
    raw_rate_b = outer.omega1_b + omega2_b
    omega_d_b = combine_and_saturate_rates(
        outer.omega1_b,
        omega2_b,
        omega_max_rad_s,
    )

    gravity = np.array((0.0, 0.0, -config.gravity_m_s2))
    desired_force = (
        config.mass_kg * (outer.acceleration_d_e - gravity) - drag
    )
    raw_thrust_n = float(rotation[:, 2] @ desired_force)
    thrust_n = min(
        max(raw_thrust_n, minimum_thrust_n),
        config.thrust_max_n,
    )
    return InnerLoopResult(
        omega2_b=omega2_b,
        omega_d_b=omega_d_b,
        thrust_n=thrust_n,
        thrust_saturated=not math.isclose(thrust_n, raw_thrust_n),
        rate_saturated=bool(
            np.linalg.norm(raw_rate_b) > omega_max_rad_s
        ),
    )


def _limit_acceleration(
    acceleration_e: Sequence[float],
    maximum_m_s2: Optional[float],
    maximum_vertical_m_s2: Optional[float],
) -> np.ndarray:
    """Bound acceleration without changing the paper command direction.

    Section III-D notes that practical saturation may change command
    magnitude, but not direction.  In particular, clipping only the vertical
    component destroys the collinearity correction contained in equation
    (19), so both envelopes are applied through one common scale factor.
    """
    acceleration = _vector3(acceleration_e, 'acceleration_d_e')
    scale = 1.0
    if maximum_vertical_m_s2 is not None:
        vertical_magnitude = abs(float(acceleration[2]))
        if vertical_magnitude > maximum_vertical_m_s2:
            scale = min(scale, maximum_vertical_m_s2 / vertical_magnitude)
    if maximum_m_s2 is not None:
        norm = float(np.linalg.norm(acceleration))
        if norm > maximum_m_s2:
            scale = min(scale, maximum_m_s2 / norm)
    return acceleration * scale


def image_los_in_earth(
    image_xy: Sequence[float],
    attitude_b_to_e: Sequence[Sequence[float]],
    camera_to_body_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Calculate ``n_t`` from normalized image coordinates using Eq. (4)."""
    image = np.asarray(image_xy, dtype=float)
    if image.shape != (2,) or not np.all(np.isfinite(image)):
        raise ValueError('image_xy must contain two finite values')
    rotation = _rotation3(attitude_b_to_e, 'attitude_b_to_e')
    camera_to_body = _rotation3(
        camera_to_body_rotation,
        'camera_to_body_rotation',
    )
    los_c = _unit3((image[0], image[1], 1.0), 'image LOS')
    return rotation @ camera_to_body @ los_c


def attitude_rate_feedback(
    attitude_d_b_to_e: Sequence[Sequence[float]],
    attitude_b_to_e: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return paper equation (26), the body-frame attitude feedback rate."""
    desired = _rotation3(attitude_d_b_to_e, 'attitude_d_b_to_e')
    actual = _rotation3(attitude_b_to_e, 'attitude_b_to_e')
    error_skew = desired.T @ actual - actual.T @ desired
    return -vex(error_skew)


def combine_and_saturate_rates(
    omega1_b: Sequence[float],
    omega2_b: Sequence[float],
    omega_max_rad_s: float,
) -> np.ndarray:
    """Apply norm-preserving rate saturation from equations (28)-(29)."""
    if not math.isfinite(omega_max_rad_s) or omega_max_rad_s <= 0.0:
        raise ValueError('omega_max_rad_s must be finite and positive')
    combined = _vector3(omega1_b, 'omega1_b') + _vector3(
        omega2_b,
        'omega2_b',
    )
    norm = float(np.linalg.norm(combined))
    if norm <= omega_max_rad_s:
        return combined
    return combined * (omega_max_rad_s / norm)


def low_pass_rates(
    previous_omega_b: Sequence[float],
    desired_omega_b: Sequence[float],
    elapsed_s: float,
    time_constant_s: float,
) -> np.ndarray:
    """Apply a time-step-independent first-order filter to body rates."""
    previous = _vector3(previous_omega_b, 'previous_omega_b')
    desired = _vector3(desired_omega_b, 'desired_omega_b')
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError('elapsed_s must be finite and nonnegative')
    if not math.isfinite(time_constant_s) or time_constant_s < 0.0:
        raise ValueError('time_constant_s must be finite and nonnegative')
    if time_constant_s == 0.0 or elapsed_s == 0.0:
        return desired.copy() if time_constant_s == 0.0 else previous.copy()
    alpha = -math.expm1(-elapsed_s / time_constant_s)
    return previous + alpha * (desired - previous)


def protect_tilt_rate(
    desired_omega_b: Sequence[float],
    attitude_b_to_e: Sequence[Sequence[float]],
    recovery_start_rad: float,
    recovery_full_rad: float,
    omega_max_rad_s: float,
) -> np.ndarray:
    """Blend a nominal paper rate into level recovery outside its envelope.

    The paper command is returned unchanged below ``recovery_start_rad``.
    Above that soft boundary, the shortest SO(3) rotation that brings the
    thrust axis back to earth-up progressively replaces the nominal command.
    This is an airframe protection layer, not a replacement control law.
    """
    desired = _vector3(desired_omega_b, 'desired_omega_b')
    rotation = _rotation3(attitude_b_to_e, 'attitude_b_to_e')
    values = (recovery_start_rad, recovery_full_rad, omega_max_rad_s)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError('tilt recovery limits and omega_max must be positive')
    if not recovery_start_rad < recovery_full_rad < math.pi / 2.0:
        raise ValueError(
            'tilt recovery limits must satisfy 0 < start < full < pi/2'
        )

    thrust_axis_e = rotation[:, 2]
    tilt_rad = math.acos(
        float(np.clip(thrust_axis_e[2], -1.0, 1.0))
    )
    if tilt_rad <= recovery_start_rad:
        return desired.copy()

    level_attitude = (
        rotation_between(thrust_axis_e, (0.0, 0.0, 1.0)) @ rotation
    )
    recovery_rate = attitude_rate_feedback(level_attitude, rotation)
    blend = min(
        1.0,
        (tilt_rad - recovery_start_rad)
        / (recovery_full_rad - recovery_start_rad),
    )
    protected = (1.0 - blend) * desired + blend * recovery_rate
    norm = float(np.linalg.norm(protected))
    if norm > omega_max_rad_s:
        protected *= omega_max_rad_s / norm
    return protected


def rotation_between(
    current_direction: Sequence[float],
    desired_direction: Sequence[float],
) -> np.ndarray:
    """Map one unit direction to another with the shortest SO(3) rotation."""
    current = _unit3(current_direction, 'current_direction')
    desired = _unit3(desired_direction, 'desired_direction')
    cosine = float(np.clip(current @ desired, -1.0, 1.0))
    cross = np.cross(current, desired)
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        axis_seed = np.zeros(3)
        axis_seed[int(np.argmin(np.abs(current)))] = 1.0
        axis = _unit3(np.cross(current, axis_seed), 'antiparallel axis')
        return rodrigues(axis, math.pi)
    return rodrigues(cross / sine, math.atan2(sine, cosine))


def rodrigues(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Convert a unit rotation axis and angle to an SO(3) matrix."""
    unit_axis = _unit3(axis, 'axis')
    if not math.isfinite(angle_rad):
        raise ValueError('angle_rad must be finite')
    axis_skew = skew(unit_axis)
    return (
        np.eye(3)
        + axis_skew * math.sin(angle_rad)
        + axis_skew @ axis_skew * (1.0 - math.cos(angle_rad))
    )


def skew(vector: Sequence[float]) -> np.ndarray:
    """Return the cross-product matrix of a three-vector."""
    x, y, z = _vector3(vector, 'vector')
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _limit_tilt_direction(
    direction: Sequence[float],
    max_tilt_rad: float,
) -> np.ndarray:
    """Limit a desired thrust direction to a realizable upward cone."""
    unit = _unit3(direction, 'direction')
    if (
        not math.isfinite(max_tilt_rad)
        or max_tilt_rad <= 0.0
        or max_tilt_rad >= math.pi / 2.0
    ):
        raise ValueError('max_tilt_rad must be within (0, pi/2)')
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


def vex(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the vector associated with a skew-symmetric matrix."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError('matrix must be a finite 3x3 matrix')
    if not np.allclose(value + value.T, 0.0, atol=1e-7):
        raise ValueError('matrix must be skew-symmetric')
    return np.array((value[2, 1], value[0, 2], value[1, 0]))


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


def _rotation3(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite 3x3 matrix')
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-7):
        raise ValueError(f'{name} must be orthonormal')
    if not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=1e-7):
        raise ValueError(f'{name} must have determinant +1')
    return value
