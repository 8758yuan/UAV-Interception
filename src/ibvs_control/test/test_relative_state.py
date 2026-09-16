import math

import pytest

from ibvs_control.relative_state import (
    ClosestApproachTracker,
    compute_relative_kinematics,
)


def test_relative_state_uses_paper_direction_convention() -> None:
    """p_r points target-to-interceptor while LOS points to the target."""
    state = compute_relative_kinematics(
        (1.0, 2.0, 3.0),
        (2.0, 0.0, 0.0),
        (11.0, 2.0, 3.0),
        (0.5, 0.0, 0.0),
    )
    assert state.p_r == (-10.0, 0.0, 0.0)
    assert state.v_r == (1.5, 0.0, 0.0)
    assert state.los == (1.0, 0.0, 0.0)
    assert state.distance_m == 10.0


def test_relative_velocity_matches_position_derivative() -> None:
    """Finite differencing p_r yields v_I minus v_T."""
    dt_s = 0.01
    p_i = (1.0, -2.0, 3.0)
    v_i = (2.0, 3.0, -1.0)
    p_t = (7.0, 5.0, 4.0)
    v_t = (-1.0, 0.5, 2.0)
    first = compute_relative_kinematics(p_i, v_i, p_t, v_t)
    second = compute_relative_kinematics(
        tuple(p_i[i] + v_i[i] * dt_s for i in range(3)),
        v_i,
        tuple(p_t[i] + v_t[i] * dt_s for i in range(3)),
        v_t,
    )
    derivative = tuple(
        (second.p_r[i] - first.p_r[i]) / dt_s for i in range(3)
    )
    assert derivative == pytest.approx(first.v_r)
    assert math.sqrt(sum(value * value for value in first.los)) == pytest.approx(1.0)


def test_zero_distance_and_nonfinite_inputs_are_rejected() -> None:
    """Undefined LOS and invalid telemetry never enter flight calculations."""
    with pytest.raises(ValueError):
        compute_relative_kinematics((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        compute_relative_kinematics((math.nan, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0))


def test_closest_approach_records_first_hit_and_minimum() -> None:
    """P2 trial metrics preserve first hit time and closest approach."""
    tracker = ClosestApproachTracker(hit_radius_m=0.5)
    tracker.update(0.0, 2.0)
    tracker.update(1.0, 0.4)
    result = tracker.update(2.0, 0.2)
    assert result.hit
    assert result.intercept_time_s == 1.0
    assert result.closest_time_s == 2.0
    assert result.d_min_m == 0.2


def test_closest_approach_rejects_time_reversal() -> None:
    """A simulation clock reset must clear the tracker before reuse."""
    tracker = ClosestApproachTracker()
    tracker.update(2.0, 1.0)
    with pytest.raises(ValueError):
        tracker.update(1.0, 0.5)
