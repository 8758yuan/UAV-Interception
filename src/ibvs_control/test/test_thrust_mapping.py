"""Tests for the P1-calibrated PX4 thrust mapping."""

import math

import pytest

from ibvs_control.thrust_mapping import (
    ThrustMappingConfig,
    newtons_to_px4_normalized,
    px4_normalized_to_newtons,
)


CONFIG = ThrustMappingConfig(
    mass_kg=2.0,
    hover_thrust_normalized=0.727,
)


def test_hover_weight_maps_to_measured_command() -> None:
    """The calibration point is reproduced exactly."""
    result = newtons_to_px4_normalized(2.0 * 9.80665, CONFIG)
    assert result.magnitude == pytest.approx(0.727)
    assert result.body_z == pytest.approx(-0.727)
    assert not result.saturated


def test_mapping_round_trip() -> None:
    """Unsaturated commands preserve their physical thrust."""
    thrust_n = 12.3
    mapped = newtons_to_px4_normalized(thrust_n, CONFIG)
    assert px4_normalized_to_newtons(mapped.magnitude, CONFIG) == pytest.approx(
        thrust_n
    )


def test_mapping_clips_above_full_command() -> None:
    """An infeasible physical request is visible as command saturation."""
    result = newtons_to_px4_normalized(
        CONFIG.thrust_at_full_command_n + 1.0,
        CONFIG,
    )
    assert result.magnitude == 1.0
    assert result.body_z == -1.0
    assert result.saturated


@pytest.mark.parametrize('value', [-1.0, math.inf, math.nan])
def test_invalid_newton_inputs_are_rejected(value: float) -> None:
    """Invalid controller output never becomes a PX4 command."""
    with pytest.raises(ValueError):
        newtons_to_px4_normalized(value, CONFIG)


@pytest.mark.parametrize('value', [-0.1, 1.1, math.inf, math.nan])
def test_invalid_normalized_inputs_are_rejected(value: float) -> None:
    """Only the documented normalized magnitude range is accepted."""
    with pytest.raises(ValueError):
        px4_normalized_to_newtons(value, CONFIG)


def test_velocity_motor_mapping_matches_paper_twr_calibration() -> None:
    config = ThrustMappingConfig(
        mass_kg=2.1143076923,
        hover_thrust_normalized=0.5219057941,
        maximum_thrust_n=3.0 * 2.0 * 9.80665,
        thrust_curve_exponent=2.0,
        actuator_minimum_fraction=0.15,
    )
    weight = config.mass_kg * config.gravity_m_s2
    hover = newtons_to_px4_normalized(weight, config)
    expected = (
        math.sqrt(weight / config.maximum_thrust_n) - 0.15
    ) / 0.85
    assert hover.magnitude == pytest.approx(expected)
    assert hover.magnitude == pytest.approx(config.hover_thrust_normalized)
    assert px4_normalized_to_newtons(hover.magnitude, config) == pytest.approx(
        weight
    )


def test_velocity_motor_mapping_includes_px4_idle_offset() -> None:
    config = ThrustMappingConfig(
        mass_kg=2.1143076923,
        hover_thrust_normalized=0.5219057941,
        maximum_thrust_n=58.8399,
        thrust_curve_exponent=2.0,
        actuator_minimum_fraction=0.15,
    )
    mapped = newtons_to_px4_normalized(15.45, config)
    expected = (math.sqrt(15.45 / 58.8399) - 0.15) / 0.85
    assert mapped.magnitude == pytest.approx(expected)
    assert mapped.magnitude < math.sqrt(15.45 / 58.8399)
    assert px4_normalized_to_newtons(mapped.magnitude, config) == pytest.approx(
        15.45
    )
