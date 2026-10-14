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
        v_r_e=(1.0, 0.0, 0.0),
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
    reduced = OuterLoopConfig(0.05, 2.5, 0.06, 2.0, 40.0, position_gain=0.3)
    original = OuterLoopConfig(0.05, 2.5, 0.06, 2.0, 40.0)
    args = ((-23.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.eye(3))
    slower = compute_outer_loop(*args, reduced)
    baseline = compute_outer_loop(*args, original)
    assert slower.acceleration_d_e[0] == pytest.approx(9.775)
    assert baseline.acceleration_d_e[0] == pytest.approx(25.875)
    assert slower.omega1_b == pytest.approx(baseline.omega1_b)


def test_position_gain_must_be_positive() -> None:
    with pytest.raises(ValueError, match='position_gain'):
        OuterLoopConfig(0.05, 2.5, 0.06, 2.0, 40.0,
                        position_gain=0.0).validate()


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


def test_radial_original_reference_recovers_full_controller_outputs() -> None:
    """Restoring k1*d requires its derivative, even with transverse motion."""
    rng = np.random.default_rng(20261014)
    for _ in range(50):
        p_r = rng.normal(size=3) * 10.0
        v_r = rng.normal(size=3) * 2.0
        n = -p_r / np.linalg.norm(p_r)
        # Optical LOS differs from geometry, as with a displaced camera.
        los = n + (0.01, -0.02, 0.0)
        args = (p_r, v_r, los, n, np.eye(3), CONFIG)
        baseline = compute_outer_loop(*args)
        restored = compute_outer_loop(
            *args,
            approach_speed_m_s=CONFIG.k1 * np.linalg.norm(p_r),
            approach_acceleration_m_s2=-CONFIG.k1 * float(n @ v_r),
        )
        for name in vars(baseline):
            assert getattr(restored, name) == pytest.approx(
                getattr(baseline, name), abs=1e-12
            )
        baseline_inner = compute_inner_loop(baseline, np.eye(3), CONFIG, 0.8)
        restored_inner = compute_inner_loop(restored, np.eye(3), CONFIG, 0.8)
        for name in vars(baseline_inner):
            assert getattr(restored_inner, name) == pytest.approx(
                getattr(baseline_inner, name), abs=1e-12
            )


def test_planned_velocity_derivative_includes_changing_target_direction() -> None:
    """Include v_c*n_dot when the geometric direction moves."""
    p_r = np.array((-3.0, 0.4, -0.2))
    v_r = np.array((0.7, 0.9, -0.3))
    n = -p_r / np.linalg.norm(p_r)
    speed, acceleration = 1.2, 0.3
    result = compute_outer_loop(
        p_r, v_r, n, n, np.eye(3), CONFIG,
        approach_speed_m_s=speed,
        approach_acceleration_m_s2=acceleration,
    )
    epsilon = 1e-6

    def desired_velocity(t):
        position = p_r + t * v_r
        return (
            -(speed + t * acceleration) * position / np.linalg.norm(position)
        )

    derivative = (
        desired_velocity(epsilon) - desired_velocity(-epsilon)
    ) / (2.0 * epsilon)
    assert result.relative_acceleration_d_e == pytest.approx(
        derivative, abs=1e-9
    )
    assert result.acceleration_d_e == pytest.approx(
        derivative - CONFIG.k2 * (v_r - speed * n) - CONFIG.position_gain * p_r
    )


def test_planning_changes_force_and_rates_without_changing_visual_feedback() -> None:
    """Pass the radial reference through the existing SO(3) and thrust chain."""
    attitude = rodrigues((0.0, 1.0, 0.0), 0.08)
    los = (math.cos(0.03), math.sin(0.03), 0.0)
    args = ((-3.0, 0.0, 0.0), (0.15, 0.0, 0.0), los,
            (1.0, 0.0, 0.0), attitude, CONFIG)
    baseline = compute_outer_loop(*args)
    planned = compute_outer_loop(
        *args, approach_speed_m_s=0.8, approach_acceleration_m_s2=0.1
    )
    assert planned.omega1_b == pytest.approx(baseline.omega1_b)
    assert not np.allclose(planned.acceleration_d_e, baseline.acceleration_d_e)
    assert not np.allclose(
        planned.attitude_d_b_to_e, baseline.attitude_d_b_to_e
    )
    assert planned.thrust_n != pytest.approx(baseline.thrust_n)
    baseline_inner = compute_inner_loop(baseline, attitude, CONFIG, 0.8)
    planned_inner = compute_inner_loop(planned, attitude, CONFIG, 0.8)
    assert not np.allclose(planned_inner.omega_d_b, baseline_inner.omega_d_b)


@pytest.mark.parametrize('reference', [
    {'approach_speed_m_s': 1.0},
    {'approach_acceleration_m_s2': 0.0},
    {'approach_speed_m_s': math.nan, 'approach_acceleration_m_s2': 0.0},
])
def test_incomplete_or_nonfinite_radial_reference_is_rejected(reference) -> None:
    with pytest.raises(ValueError, match='both be finite'):
        compute_outer_loop(
            (-3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), np.eye(3), CONFIG, **reference,
        )
