from dataclasses import replace
import math

import numpy as np
import pytest

from ibvs_control.so3_controller import (
    OuterLoopConfig,
    attitude_rate_feedback,
    combine_and_saturate_rates,
    compute_drag_force_e,
    compute_inner_loop,
    compute_outer_loop,
    image_los_in_earth,
    rodrigues,
    rotation_between,
    skew,
    vex,
)


CONFIG = OuterLoopConfig(
    k1=0.1,
    k2=0.2,
    k_b=0.3,
    mass_kg=2.0,
    thrust_max_n=40.0,
)


def test_skew_and_vex_are_inverse_cross_product_maps() -> None:
    vector = np.array((1.0, -2.0, 3.0))
    other = np.array((4.0, 5.0, -6.0))
    assert skew(vector) @ other == pytest.approx(np.cross(vector, other))
    assert vex(skew(vector)) == pytest.approx(vector)


def test_paper_drag_force_opposes_earth_velocity() -> None:
    drag = compute_drag_force_e(
        velocity_e=(1.0, -2.0, 3.0),
        attitude_b_to_e=np.eye(3),
        drag_coefficients_b=(0.1, 0.2, 0.3),
    )
    assert drag == pytest.approx((-0.1, 0.4, -0.9))


def test_paper_drag_force_rotates_body_coefficients_into_earth() -> None:
    attitude = rodrigues((0.0, 0.0, 1.0), math.pi / 2.0)
    drag = compute_drag_force_e(
        velocity_e=(1.0, 0.0, 0.0),
        attitude_b_to_e=attitude,
        drag_coefficients_b=(1.0, 2.0, 3.0),
    )
    assert drag == pytest.approx((-2.0, 0.0, 0.0))


def test_paper_drag_rejects_negative_coefficients() -> None:
    with pytest.raises(ValueError, match='nonnegative'):
        compute_drag_force_e(
            velocity_e=(1.0, 0.0, 0.0),
            attitude_b_to_e=np.eye(3),
            drag_coefficients_b=(-0.1, 0.2, 0.3),
        )


def test_rotation_between_maps_current_thrust_to_desired_thrust() -> None:
    current = np.array((0.0, 0.0, 1.0))
    desired = np.array((0.8, 0.0, 0.6))
    rotation = rotation_between(current, desired)
    assert rotation @ current == pytest.approx(desired)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_antiparallel_thrust_rotation_is_well_defined() -> None:
    rotation = rotation_between((0, 0, 1), (0, 0, -1))
    assert rotation @ np.array((0, 0, 1)) == pytest.approx((0, 0, -1))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_aligned_outer_loop_has_zero_los_rate() -> None:
    result = compute_outer_loop(
        p_r_e=(-10.0, 0.0, 0.0),
        v_r_e=(CONFIG.k1 * 10.0 + CONFIG.v_c, 0.0, 0.0),
        los_e=(1.0, 0.0, 0.0),
        designed_los_e=(1.0, 0.0, 0.0),
        attitude_b_to_e=np.eye(3),
        config=CONFIG,
    )
    assert result.z1 == pytest.approx(0.0)
    assert result.z2_e == pytest.approx((0.0, 0.0, 0.0))
    assert result.omega1_b == pytest.approx((0.0, 0.0, 0.0))
    assert result.attitude_d_b_to_e[:, 2] == pytest.approx(
        result.thrust_direction_d_e
    )


def test_position_gain_reduces_far_range_acceleration() -> None:
    """A smaller position gain prevents large initial acceleration demand."""
    reduced = OuterLoopConfig(
        0.05, 2.5, 0.06, 2.0, 40.0, position_gain=(0.3, 0.3, 0.3)
    )
    original = OuterLoopConfig(0.05, 2.5, 0.06, 2.0, 40.0)
    args = ((-23.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.eye(3))
    slower = compute_outer_loop(*args, reduced)
    baseline = compute_outer_loop(*args, original)
    assert slower.acceleration_d_e[0] == pytest.approx(
        9.775 + reduced.k2 * reduced.v_c
    )
    assert baseline.acceleration_d_e[0] == pytest.approx(
        25.875 + original.k2 * original.v_c
    )
    assert slower.omega1_b == pytest.approx(baseline.omega1_b)


@pytest.mark.parametrize('gain', (
    (0.0, 0.8, 0.8), (0.8, -0.1, 0.8), (0.8, 0.8, math.nan),
    (math.inf, 0.8, 0.8), (0.8, 0.8), (0.8, 0.8, 0.8, 0.8), 0.8,
))
def test_position_gain_requires_three_finite_positive_components(gain) -> None:
    with pytest.raises(ValueError, match='position_gain'):
        replace(CONFIG, position_gain=gain).validate()


@pytest.mark.parametrize('axis', range(3))
def test_position_gain_changes_only_its_enu_acceleration_component(axis) -> None:
    position = np.array((-3.0, -4.0, -2.0))
    baseline_config = replace(CONFIG, position_gain=(0.8, 0.8, 0.8))
    gains = [0.8, 0.8, 0.8]
    gains[axis] = 0.2
    args = (position, (0.2, -0.1, 0.3), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3))
    baseline = compute_outer_loop(*args, baseline_config)
    adjusted = compute_outer_loop(
        *args, replace(baseline_config, position_gain=tuple(gains))
    )
    expected_delta = np.zeros(3)
    expected_delta[axis] = (0.8 - 0.2) * position[axis]
    assert adjusted.acceleration_d_e - baseline.acceleration_d_e == (
        pytest.approx(expected_delta)
    )
    assert adjusted.z2_e == pytest.approx(baseline.z2_e)
    assert adjusted.omega1_b == pytest.approx(baseline.omega1_b)


@pytest.mark.parametrize('v_c', (0.0, -0.1, math.nan, math.inf))
def test_closing_speed_must_be_finite_and_positive(v_c) -> None:
    with pytest.raises(ValueError, match='v_c'):
        replace(CONFIG, v_c=v_c).validate()


@pytest.mark.parametrize('distance', (10.0, 0.1, 1e-5))
def test_radial_reference_keeps_constant_closing_component(distance) -> None:
    velocity = np.array((0.3, 0.0, 0.0))
    outer = compute_outer_loop(
        (-distance, 0.0, 0.0), velocity, (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0), np.eye(3), CONFIG,
    )
    assert velocity - outer.z2_e == pytest.approx(
        (CONFIG.k1 * distance + CONFIG.v_c, 0.0, 0.0)
    )
    assert outer.acceleration_d_e == pytest.approx(
        -CONFIG.k1 * velocity - CONFIG.k2 * outer.z2_e
        + np.array((CONFIG.position_gain[0] * distance, 0.0, 0.0))
    )


@pytest.mark.parametrize('v_c', (0.25, 1.7))
def test_reference_derivative_matches_finite_difference_with_lateral_motion(
    v_c,
) -> None:
    config = replace(CONFIG, v_c=v_c)
    rng = np.random.default_rng(20261019)
    # Deliberately different image and geometric bearings: the image remains
    # authoritative for the IBVS term, while the requested reference uses p_r.
    los = np.array((math.cos(0.08), math.sin(0.08), 0.0))
    designed = np.array((1.0, 0.0, 0.0))
    target_acceleration = np.array((0.1, -0.2, 0.05))
    for _ in range(32):
        position = rng.normal(size=3)
        position *= rng.uniform(0.2, 20.0) / np.linalg.norm(position)
        velocity = rng.normal(size=3)

        def reference(p):
            return -(config.k1 + v_c / np.linalg.norm(p)) * p

        step = 1e-6
        numerical_derivative = (
            reference(position + step * velocity)
            - reference(position - step * velocity)
        ) / (2.0 * step)
        outer = compute_outer_loop(
            position, velocity, los, designed, np.eye(3), config,
            target_acceleration_e=target_acceleration,
        )
        expected_z2 = velocity - reference(position)
        barrier = outer.z1 / (config.k_b**2 - outer.z1**2)
        visual_acceleration = (
            barrier * config.mass_kg / np.linalg.norm(position)
            * (-np.eye(3) + np.outer(los, los)) @ designed
        )
        assert outer.z2_e == pytest.approx(expected_z2)
        assert outer.acceleration_d_e == pytest.approx(
            numerical_derivative - config.k2 * expected_z2
            - np.asarray(config.position_gain) * position + visual_acceleration
            + target_acceleration,
            abs=1e-8,
        )


@pytest.mark.parametrize('mass', (1.0, 2.3))
@pytest.mark.parametrize('position_gain', ((0.8, 0.8, 0.8), (0.2, 0.4, 0.8)))
def test_weighted_lyapunov_identity_includes_full_loop_residuals(
    mass, position_gain,
) -> None:
    config = replace(CONFIG, mass_kg=mass, position_gain=position_gain)
    gains = np.asarray(position_gain)
    position = np.array((-3.0, -0.4, -0.2))
    velocity = np.array((0.8, -0.3, 0.2))
    distance = np.linalg.norm(position)
    position_quadratic = position @ (gains * position)
    los = -position / distance
    projection = np.eye(3) - np.outer(los, los)
    rotation = rodrigues((0.0, 1.0, 0.0), 0.2)
    designed = rotation @ np.array((1.0, 0.0, 0.0))
    drag = np.array((-0.1, 0.05, 0.0))
    outer = compute_outer_loop(
        position, velocity, los, designed, rotation, config,
        drag_force_e=drag,
    )
    inner = compute_inner_loop(outer, rotation, config, 0.2, drag_force_e=drag)
    barrier = outer.z1 / (config.k_b**2 - outer.z1**2)
    cross = np.cross(designed, los)
    attitude_error = -inner.omega2_b
    w = outer.omega1_b + inner.omega2_b
    rate_error = np.array((0.01, -0.02, 0.03))
    actual_rate = inner.omega_d_b + rate_error
    desired_rate = np.array((0.02, -0.04, 0.01))
    desired_rate_b = rotation.T @ outer.attitude_d_b_to_e @ desired_rate
    actual_acceleration = (
        np.array((0.0, 0.0, -config.gravity_m_s2))
        + inner.thrust_n / mass * rotation[:, 2] + drag / mass
    )
    acceleration_error = actual_acceleration - outer.acceleration_d_e
    reference_dot = (
        -config.k1 * velocity - config.v_c / distance * projection @ velocity
    )
    barrier_dot = barrier * (
        designed @ projection @ velocity / distance
        - cross @ rotation @ actual_rate
    )
    actual_v_dot = (
        mass * barrier_dot + (gains * position) @ velocity
        + outer.z2_e @ (actual_acceleration - reference_dot)
        + mass * attitude_error @ (actual_rate - desired_rate_b)
    )
    expected_v_dot = (
        -(config.k1 + config.v_c / distance) * position_quadratic
        - config.k2 * (outer.z2_e @ outer.z2_e)
        - mass * w @ inner.omega_d_b
        + outer.z2_e @ acceleration_error - mass * w @ rate_error
        - mass * attitude_error @ desired_rate_b
    )
    assert actual_v_dot == pytest.approx(expected_v_dot, abs=1e-10)

    def candidate(p, v, r, rd):
        d = np.linalg.norm(p)
        n = -p / d
        z1 = 1.0 - (r @ np.array((1.0, 0.0, 0.0))) @ n
        z2 = v - (config.k1 * d + config.v_c) * n
        return (
            mass / 2.0 * math.log(config.k_b**2 / (config.k_b**2 - z1**2))
            + 0.5 * p @ (gains * p) + (z2 @ z2) / 2.0
            + mass * np.trace(np.eye(3) - rd.T @ r)
        )

    step = 1e-6
    values = []
    for h in (-step, step):
        r = rotation @ rodrigues(actual_rate, h * np.linalg.norm(actual_rate))
        rd = outer.attitude_d_b_to_e @ rodrigues(
            desired_rate, h * np.linalg.norm(desired_rate)
        )
        values.append(candidate(
            position + h * velocity, velocity + h * actual_acceleration, r, rd,
        ))
    assert (values[1] - values[0]) / (2.0 * step) == pytest.approx(
        expected_v_dot, abs=1e-7,
    )


def test_los_rate_rotates_designed_los_toward_target() -> None:
    angle = math.radians(10.0)
    target_los = (math.cos(angle), math.sin(angle), 0.0)
    result = compute_outer_loop(
        (-10.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        target_los,
        (1.0, 0.0, 0.0),
        np.eye(3),
        CONFIG,
    )
    assert result.omega1_b[2] > 0.0


def test_barrier_violation_is_rejected_instead_of_clamped() -> None:
    with pytest.raises(ValueError, match='barrier'):
        compute_outer_loop(
            (-10.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            np.eye(3),
            CONFIG,
        )


def test_attitude_feedback_is_zero_at_desired_attitude() -> None:
    assert attitude_rate_feedback(np.eye(3), np.eye(3)) == pytest.approx(
        np.zeros(3)
    )
    desired = rodrigues((1.0, 0.0, 0.0), math.radians(10.0))
    feedback = attitude_rate_feedback(desired, np.eye(3))
    assert feedback[0] > 0.0


def test_rate_saturation_preserves_direction_and_limits_norm() -> None:
    result = combine_and_saturate_rates((3, 4, 0), (0, 0, 0), 2.0)
    assert np.linalg.norm(result) == pytest.approx(2.0)
    assert result == pytest.approx((1.2, 1.6, 0.0))


def test_thrust_is_clipped_without_changing_desired_direction() -> None:
    low_limit = OuterLoopConfig(0.1, 0.2, 0.3, 2.0, 1.0)
    result = compute_outer_loop(
        (-10.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        np.eye(3),
        low_limit,
    )
    assert result.thrust_n == 1.0
    assert result.thrust_saturated
    assert np.linalg.norm(result.thrust_direction_d_e) == pytest.approx(1.0)


def test_image_los_matches_paper_equation_4_and_camera_mount() -> None:
    camera_to_body = np.array(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )
    los = image_los_in_earth((0.2, -0.1), np.eye(3), camera_to_body)
    assert los == pytest.approx(
        np.array((1.0, -0.2, 0.1)) / np.linalg.norm((1.0, -0.2, 0.1))
    )


def test_inner_loop_recomputes_equations_23_26_and_28() -> None:
    outer = compute_outer_loop(
        (-10.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        np.eye(3),
        CONFIG,
    )
    current = rodrigues((0.0, 1.0, 0.0), math.radians(5.0))
    inner = compute_inner_loop(outer, current, CONFIG, 0.1)
    assert inner.omega2_b == pytest.approx(
        attitude_rate_feedback(outer.attitude_d_b_to_e, current)
    )
    assert np.linalg.norm(inner.omega_d_b) == pytest.approx(0.1)
    assert inner.rate_saturated
    assert 0.0 <= inner.thrust_n <= CONFIG.thrust_max_n
