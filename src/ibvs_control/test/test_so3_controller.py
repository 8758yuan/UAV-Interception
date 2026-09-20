import math

import numpy as np
import pytest

from ibvs_control.so3_controller import (
    OuterLoopConfig,
    attitude_rate_feedback,
    combine_and_saturate_rates,
    compute_inner_loop,
    compute_outer_loop,
    image_los_in_earth,
    low_pass_rates,
    protect_tilt_rate,
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


def test_rate_low_pass_is_time_step_independent_and_bounded() -> None:
    one_step = low_pass_rates((0, 0, 0), (1, -1, 0), 0.1, 0.2)
    two_steps = low_pass_rates((0, 0, 0), (1, -1, 0), 0.05, 0.2)
    two_steps = low_pass_rates(two_steps, (1, -1, 0), 0.05, 0.2)
    assert two_steps == pytest.approx(one_step)
    assert np.all(np.abs(one_step) < 1.0)


def test_zero_time_constant_disables_rate_filter() -> None:
    assert low_pass_rates((1, 2, 3), (-1, -2, -3), 0.01, 0.0) == pytest.approx(
        (-1, -2, -3)
    )


def test_tilt_protection_is_inactive_inside_nominal_envelope() -> None:
    desired = np.array((0.1, -0.05, 0.02))
    protected = protect_tilt_rate(
        desired,
        rodrigues((0.0, 1.0, 0.0), math.radians(8.0)),
        math.radians(10.0),
        math.radians(15.0),
        0.25,
    )
    assert protected == pytest.approx(desired)


def test_tilt_protection_commands_level_recovery_at_full_boundary() -> None:
    attitude = rodrigues((0.0, 1.0, 0.0), math.radians(15.0))
    protected = protect_tilt_rate(
        (0.0, 0.25, 0.0),
        attitude,
        math.radians(10.0),
        math.radians(15.0),
        0.25,
    )
    # Positive body-Y would increase this pitch; protection must reverse it.
    assert protected[1] < 0.0
    assert np.linalg.norm(protected) <= 0.25 + 1e-12


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


def test_vehicle_acceleration_envelope_bounds_ideal_paper_command() -> None:
    bounded = OuterLoopConfig(
        k1=0.1,
        k2=0.2,
        k_b=0.3,
        mass_kg=2.0,
        thrust_max_n=40.0,
        max_acceleration_m_s2=6.0,
        max_vertical_acceleration_m_s2=2.0,
    )
    result = compute_outer_loop(
        (-20.0, 3.0, -8.0),
        (4.0, -2.0, 3.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        np.eye(3),
        bounded,
    )
    assert np.linalg.norm(result.acceleration_d_e) <= 6.0 + 1e-12
    assert abs(result.acceleration_d_e[2]) <= 2.0


def test_minimum_thrust_prevents_a_cut_during_interception() -> None:
    outer = compute_outer_loop(
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        np.eye(3),
        CONFIG,
    )
    result = compute_inner_loop(
        outer,
        np.eye(3),
        CONFIG,
        0.5,
        minimum_thrust_n=12.0,
    )
    assert result.thrust_n == pytest.approx(12.0)
    assert result.thrust_saturated


def test_desired_thrust_direction_respects_command_tilt_limit() -> None:
    """Large force requests are projected into the realizable tilt cone."""
    result = compute_outer_loop(
        (-10.0, -10.0, -10.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        np.eye(3),
        OuterLoopConfig(
            k1=0.1,
            k2=0.2,
            k_b=0.3,
            mass_kg=2.0,
            thrust_max_n=40.0,
            max_command_tilt_rad=math.radians(20.0),
        ),
    )
    tilt = math.degrees(
        math.acos(float(result.thrust_direction_d_e[2]))
    )
    assert tilt == pytest.approx(20.0)
    assert result.attitude_d_b_to_e[:, 2] == pytest.approx(
        result.thrust_direction_d_e
    )


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
