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
