"""Tests for the final controller-to-PX4 command conversion."""

import numpy as np
import pytest

from ibvs_control.px4_command_adapter import adapt_rate_thrust_command
from ibvs_control.thrust_mapping import ThrustMappingConfig


MAPPING = ThrustMappingConfig(2.0, 0.727)


def test_flu_rates_and_thrust_convert_to_px4_frd() -> None:
    """Pitch, yaw, and thrust signs follow the PX4 FRD convention."""
    command = adapt_rate_thrust_command(
        (0.1, 0.2, -0.3),
        2.0 * 9.80665,
        0.5,
        MAPPING,
    )
    assert command.rates_frd_rad_s == pytest.approx((0.1, -0.2, 0.3))
    assert command.thrust_body == pytest.approx((0.0, 0.0, -0.727))
    assert not command.rate_saturated
    assert not command.thrust_saturated


def test_adapter_defensively_preserves_rate_direction_when_clipping() -> None:
    """The final adapter independently enforces the configured norm bound."""
    command = adapt_rate_thrust_command((1.0, 2.0, 2.0), 10.0, 0.3, MAPPING)
    assert np.linalg.norm(command.rates_frd_rad_s) == pytest.approx(0.3)
    assert command.rate_saturated


def test_adapter_reports_thrust_mapping_saturation() -> None:
    """An infeasible Newton request remains visible to the coordinator."""
    command = adapt_rate_thrust_command(
        (0.0, 0.0, 0.0),
        MAPPING.thrust_at_full_command_n + 1.0,
        0.5,
        MAPPING,
    )
    assert command.thrust_body == (0.0, 0.0, -1.0)
    assert command.thrust_saturated


@pytest.mark.parametrize(
    'rates',
    [(1.0, 2.0), (0.0, float('nan'), 0.0)],
)
def test_adapter_rejects_invalid_rates(rates) -> None:
    """Malformed controller values never become PX4 messages."""
    with pytest.raises(ValueError):
        adapt_rate_thrust_command(rates, 10.0, 0.5, MAPPING)
