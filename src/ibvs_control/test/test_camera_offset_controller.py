"""Finite-difference checks of the offset-camera control geometry."""

import math

import numpy as np
import pytest

from ibvs_control.so3_controller import (
    OuterLoopConfig,
    compute_inner_loop,
    compute_outer_loop,
    rodrigues,
)


CONFIG = OuterLoopConfig(
    k1=0.05, velocity_gain=(0.8, 1.0, 3.0),
    k_b=1.0 - math.cos(math.radians(30.0)), mass_kg=2.3,
    thrust_max_n=40.0, position_gain=(0.2, 0.4, 1.0), v_c=0.1,
)
OFFSETS = ((0.0, 0.0, 0.0), (0.14, 0.0, 0.0), (0.12, -0.03, 0.2))


def barrier(position, rotation, offset):
    """Project the physical camera and evaluate its barrier energy."""
    ray = -position - rotation @ offset
    los = ray / np.linalg.norm(ray)
    z1 = 1.0 - rotation[:, 0] @ los
    return 0.5 * math.log(CONFIG.k_b**2 / (CONFIG.k_b**2 - z1**2))


@pytest.mark.parametrize('offset', OFFSETS)
@pytest.mark.parametrize('camera_range', (0.15, 0.5, 12.0))
def test_camera_barrier_derivative_and_reference_drift(offset, camera_range):
    offset = np.asarray(offset)
    rotation = rodrigues((0.2, -0.3, 0.4), 0.35)
    los_b = np.array((1.0, 0.12, -0.07))
    los_b /= np.linalg.norm(los_b)
    los = rotation @ los_b
    position = -rotation @ (offset + camera_range * los_b)
    velocity = np.array((0.3, -0.4, 0.2))
    rate = np.array((0.5, -0.2, 0.3))
    outer = compute_outer_loop(
        position, velocity, los, rotation[:, 0], rotation, CONFIG,
        camera_position_b_m=offset,
    )
    distance = np.linalg.norm(position)
    radial = -position / distance
    reference_dot = (
        -CONFIG.k1 * velocity
        - CONFIG.v_c / distance * (velocity - radial * (radial @ velocity))
    )
    visual_acceleration = (
        outer.acceleration_d_e - reference_dot
        + np.asarray(CONFIG.velocity_gain) * outer.z2_e
        + np.asarray(CONFIG.position_gain) * position
    )
    reference_rate_e = (
        (CONFIG.k1 + CONFIG.v_c / distance)
        * np.cross(rotation @ offset, los)
        / (camera_range + (rotation @ offset) @ los)
    )
    barrier_rotation_e = rotation @ outer.omega1_b - reference_rate_e
    expected = (
        -visual_acceleration @ velocity
        / (CONFIG.mass_kg * (1.0 + camera_range / CONFIG.visual_l_m))
        - barrier_rotation_e @ rotation @ rate
    )

    def derivative(v, omega):
        step = 1e-6
        values = [barrier(
            position + h * v,
            rotation @ rodrigues(omega, h * np.linalg.norm(omega)), offset,
        ) for h in (-step, step)]
        return (values[1] - values[0]) / (2.0 * step)

    assert derivative(velocity, rate) == pytest.approx(expected, abs=1e-8)
    reference_velocity = velocity - outer.z2_e
    # Along the existing radial reference, the extra camera drift cancels.
    assert derivative(reference_velocity, outer.omega1_b) == pytest.approx(
        -barrier_rotation_e @ barrier_rotation_e, abs=1e-8,
    )


@pytest.mark.parametrize('offset', OFFSETS)
@pytest.mark.parametrize('camera_range', (0.5, 12.0))
@pytest.mark.parametrize('rate_limit', (0.05, 3.0))
def test_offset_lyapunov_identity_retains_actuator_and_attitude_residuals(
    offset, camera_range, rate_limit,
):
    offset = np.asarray(offset)
    rotation = rodrigues((0.0, 1.0, 0.0), 0.2)
    los_b = np.array((1.0, 0.15, -0.08))
    los_b /= np.linalg.norm(los_b)
    los = rotation @ los_b
    position = -rotation @ (offset + camera_range * los_b)
    velocity = np.array((0.8, -0.3, 0.2))
    drag = np.array((-0.1, 0.05, 0.0))
    outer = compute_outer_loop(
        position, velocity, los, rotation[:, 0], rotation, CONFIG,
        camera_position_b_m=offset, drag_force_e=drag,
    )
    inner = compute_inner_loop(outer, rotation, CONFIG, rate_limit,
                               drag_force_e=drag)
    mass = CONFIG.mass_kg
    acceleration = (
        np.array((0.0, 0.0, -CONFIG.gravity_m_s2))
        + inner.thrust_n / mass * rotation[:, 2] + drag / mass
    )
    rate_error = np.array((0.01, -0.02, 0.03))
    rate = inner.omega_d_b + rate_error
    desired_rate = np.array((0.02, -0.04, 0.01))
    desired_rate_b = rotation.T @ outer.attitude_d_b_to_e @ desired_rate
    reference_rate_b = (
        (CONFIG.k1 + CONFIG.v_c / np.linalg.norm(position))
        * np.cross(offset, los_b) / (camera_range + offset @ los_b)
    )
    barrier_rotation_b = outer.omega1_b - reference_rate_b
    attitude_error = -inner.omega2_b
    w = barrier_rotation_b - attitude_error
    raw_norm = np.linalg.norm(outer.omega1_b + inner.omega2_b)
    scale = min(1.0, rate_limit / raw_norm)
    expected = (
        -(CONFIG.k1 + CONFIG.v_c / np.linalg.norm(position))
        * position @ (np.asarray(CONFIG.position_gain) * position)
        - outer.z2_e @ (np.asarray(CONFIG.velocity_gain) * outer.z2_e)
        - mass * scale * (w @ w)
        + mass * ((1.0 - scale) * barrier_rotation_b
                  + scale * attitude_error) @ reference_rate_b
        + outer.z2_e @ (acceleration - outer.acceleration_d_e)
        - mass * w @ rate_error - mass * attitude_error @ desired_rate_b
        - mass * outer.z1 / (CONFIG.k_b**2 - outer.z1**2) / CONFIG.visual_l_m
        * outer.z2_e @ (np.eye(3) - np.outer(los, los)) @ rotation[:, 0]
    )

    def candidate(p, v, r, rd):
        z2 = v + (CONFIG.k1 + CONFIG.v_c / np.linalg.norm(p)) * p
        return (
            mass * barrier(p, r, offset)
            + 0.5 * p @ (np.asarray(CONFIG.position_gain) * p)
            + 0.5 * z2 @ z2 + mass * np.trace(np.eye(3) - rd.T @ r)
        )

    step = 1e-6
    values = [candidate(
        position + h * velocity, velocity + h * acceleration,
        rotation @ rodrigues(rate, h * np.linalg.norm(rate)),
        outer.attitude_d_b_to_e
        @ rodrigues(desired_rate, h * np.linalg.norm(desired_rate)),
    ) for h in (-step, step)]
    assert (values[1] - values[0]) / (2.0 * step) == pytest.approx(
        expected, abs=1e-6,
    )


def test_forward_camera_keeps_aligned_body_origin_control():
    args = ((-1.0, 0.0, 0.0), (0.4, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3), CONFIG)
    centred = compute_outer_loop(*args)
    forward = compute_outer_loop(*args, camera_position_b_m=(0.14, 0.0, 0.0))
    assert forward.acceleration_d_e == pytest.approx(centred.acceleration_d_e)
    assert forward.z2_e == pytest.approx(centred.z2_e)
    assert forward.omega1_b == pytest.approx(np.zeros(3))


@pytest.mark.parametrize('offset', ((0.14, 0.0), (math.nan, 0.0, 0.0)))
def test_camera_mount_requires_a_finite_three_vector(offset):
    with pytest.raises(ValueError, match='camera_position_b_m'):
        compute_outer_loop(
            (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3), CONFIG, camera_position_b_m=offset,
        )


def test_target_at_camera_centre_is_a_numerical_singularity():
    with pytest.raises(ValueError, match='camera distance'):
        compute_outer_loop(
            (-0.14, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3), CONFIG,
            camera_position_b_m=(0.14, 0.0, 0.0),
        )


def test_singular_reference_drift_uses_existing_invalid_control_path():
    with pytest.raises(ValueError, match='reference drift correction'):
        compute_outer_loop(
            (0.28, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3), CONFIG,
            camera_position_b_m=(-0.14, 0.0, 0.0),
        )
