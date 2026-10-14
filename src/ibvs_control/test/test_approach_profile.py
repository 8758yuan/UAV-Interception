"""Boundary, continuity and geometric checks for the V3 radial reference."""

from dataclasses import replace
import math

import numpy as np
import pytest

from ibvs_control.approach_profile import (
    ApproachPlanner,
    ApproachProfileConfig,
    ApproachReference,
    QuinticApproachSegment,
)


def test_quintic_satisfies_all_six_boundaries() -> None:
    rng = np.random.default_rng(20261014)
    for _ in range(50):
        start = ApproachReference(*rng.uniform(-5.0, 5.0, 3))
        end = ApproachReference(*rng.uniform(-5.0, 5.0, 3))
        h = rng.uniform(0.2, 4.0)
        segment = QuinticApproachSegment.from_boundaries(10.0, h, start, end)
        b = segment.coefficients
        for s, expected in ((0.0, start), (1.0, end)):
            actual = (
                np.polynomial.polynomial.polyval(s, b),
                -np.polynomial.polynomial.polyval(
                    s, np.polynomial.polynomial.polyder(b)
                ) / h,
                -np.polynomial.polynomial.polyval(
                    s, np.polynomial.polynomial.polyder(b, 2)
                ) / h**2,
            )
            assert actual == pytest.approx(
                tuple(vars(expected).values()), abs=1e-10
            )


def test_velocity_and_acceleration_are_distance_derivatives() -> None:
    segment = QuinticApproachSegment.from_boundaries(
        1.0, 2.0, ApproachReference(10.0, 1.0, 0.2),
        ApproachReference(6.0, 3.0, -0.1),
    )
    epsilon = 1e-5
    for time in (1.3, 1.8, 2.7):
        left = segment.sample(time - epsilon)
        right = segment.sample(time + epsilon)
        reference = segment.sample(time)
        assert reference.speed_m_s == pytest.approx(
            -(right.distance_m - left.distance_m) / (2.0 * epsilon)
        )
        assert reference.acceleration_m_s2 == pytest.approx(
            (right.speed_m_s - left.speed_m_s) / (2.0 * epsilon)
        )


def test_first_reference_retains_the_complete_original_law() -> None:
    planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
    reference = planner.update(0.0, (-10.0, 0.0, 0.0), (2.0, 1.0, 0.0), 0.02)
    assert reference == ApproachReference(10.0, 0.5, -0.1)


def test_target_uses_optical_error_and_geometric_transverse_velocity() -> None:
    config = ApproachProfileConfig(delta_v_m_s=1.2)
    planner = ApproachPlanner(config, k1=0.05)
    planner.update(0.0, (-10.0, 0.0, 0.0), (2.0, 3.0, 0.0), 0.03)
    nominal = (
        0.5 + config.delta_v_m_s * 10.0 / (10.0 + config.boost_length_m)
        + config.terminal_speed_m_s
        * math.exp(-10.0 / config.terminal_length_m)
    )
    assert planner.weight == pytest.approx(
        math.exp(-config.alpha * 0.03 - config.beta * 9.0)
    )
    assert planner.target_speed_m_s == pytest.approx(planner.weight * nominal)


def test_replanning_inherits_reference_instead_of_new_measurement() -> None:
    planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
    planner.update(0.0, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    segment = planner.segment
    expected = segment.sample(0.25)
    actual = planner.update(0.25, (-15.0, 2.0, 1.0), (3.0, 1.0, 0.0), 0.04)
    assert actual == expected
    assert planner.replan_count == 2
    assert planner.segment.start == expected
    assert planner.segment is not segment
    assert actual.distance_m != pytest.approx(
        np.linalg.norm((-15.0, 2.0, 1.0))
    )


def test_outer_loop_sampling_does_not_replan_each_tick() -> None:
    planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
    planner.update(0.0, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    for index in range(1, 10):
        planner.update(index * 0.02, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert planner.replan_count == 1
    planner.update(0.2, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert planner.replan_count == 2


def test_continuous_weight_has_no_visual_or_transverse_cutoff() -> None:
    speeds = []
    for z1 in (0.02 - 1e-7, 0.02, 0.02 + 1e-7):
        planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
        planner.update(0.0, (-10.0, 0.0, 0.0), (0.0, 1.0, 0.0), z1)
        speeds.append(planner.target_speed_m_s)
    assert speeds[0] > speeds[1] > speeds[2] > 0.0
    assert max(speeds) - min(speeds) < 1e-5


def test_near_target_has_nonzero_terminal_speed_and_positive_distance() -> None:
    planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
    planner.update(0.0, (-1e-4, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    segment = planner.segment
    assert planner.target_speed_m_s == pytest.approx(0.45, rel=1e-4)
    assert segment.horizon_s < planner.config.horizon_s
    for time in np.linspace(0.0, segment.horizon_s, 101):
        assert segment.sample(time).distance_m > 0.0
    end = segment.end
    # A completed segment keeps the terminal command until the next outer tick.
    actual = planner.update(0.02, (-1e-4, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert actual == end
    assert actual.speed_m_s > 0.0
    assert planner.replan_count == 2


def test_geometric_horizon_handles_inherited_speed_and_acceleration() -> None:
    rng = np.random.default_rng(20261014)
    for _ in range(50):
        planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
        start = ApproachReference(
            rng.uniform(0.001, 10.0), rng.uniform(-10.0, 10.0),
            rng.uniform(-100.0, 100.0),
        )
        planner.segment = QuinticApproachSegment.from_boundaries(
            -1.0, 1.0, start, start
        )
        planner.update(0.0, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        segment = planner.segment
        for time in np.linspace(0.0, segment.horizon_s, 51):
            assert segment.sample(time).distance_m > 0.0


def test_reset_and_clock_rewind_reinitialize_from_original_reference() -> None:
    planner = ApproachPlanner(ApproachProfileConfig(), k1=0.05)
    planner.update(10.0, (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert planner.update(1.0, (-8.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0) == (
        ApproachReference(8.0, 0.4, -0.05)
    )
    planner.reset()
    assert planner.segment is None
    assert planner.replan_count == 0


@pytest.mark.parametrize('changes', [
    {'horizon_s': 0.0}, {'beta': -1.0}, {'terminal_speed_m_s': 0.0},
    {'alpha': math.nan},
])
def test_invalid_design_parameters_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        ApproachPlanner(replace(ApproachProfileConfig(), **changes), k1=0.05)
