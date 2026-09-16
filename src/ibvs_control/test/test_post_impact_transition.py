"""Tests for the non-reversing post-impact velocity transition."""

import numpy as np
import pytest

from ibvs_control.post_impact_transition import smooth_stop_command


def test_smooth_stop_is_continuous_at_both_endpoints() -> None:
    initial = np.array((0.2, 2.0, -0.1))
    start = smooth_stop_command(initial, 0.0, 3.0)
    end = smooth_stop_command(initial, 3.0, 3.0)
    assert start.velocity_ned == pytest.approx(initial)
    assert start.acceleration_ned == pytest.approx((0.0, 0.0, 0.0))
    assert end.velocity_ned == pytest.approx((0.0, 0.0, 0.0))
    assert end.acceleration_ned == pytest.approx((0.0, 0.0, 0.0))


def test_smooth_stop_never_reverses_velocity() -> None:
    initial = np.array((-0.2, 2.0, -0.1))
    previous_norm = float(np.linalg.norm(initial))
    for elapsed_s in np.linspace(0.0, 3.0, 31):
        command = smooth_stop_command(initial, elapsed_s, 3.0)
        assert np.all(command.velocity_ned * initial >= 0.0)
        current_norm = float(np.linalg.norm(command.velocity_ned))
        assert current_norm <= previous_norm + 1e-12
        previous_norm = current_norm


def test_smooth_stop_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        smooth_stop_command((1.0, 2.0), 0.0, 3.0)
    with pytest.raises(ValueError):
        smooth_stop_command((1.0, 0.0, 0.0), -0.1, 3.0)
    with pytest.raises(ValueError):
        smooth_stop_command((1.0, 0.0, 0.0), 0.0, 0.0)
