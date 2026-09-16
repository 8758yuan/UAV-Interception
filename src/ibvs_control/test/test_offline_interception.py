import pytest

from ibvs_control.offline_interception import (
    OfflineSimulationConfig,
    default_controller_config,
    simulate_interception,
)


def test_default_candidate_hits_within_initial_p2_limits() -> None:
    """The frozen offline baseline reaches the target without limit breaches."""
    simulation = OfflineSimulationConfig()
    result = simulate_interception(default_controller_config(), simulation)
    assert result.hit
    assert result.safe
    assert result.reason == 'hit'
    assert result.d_min_m <= simulation.hit_radius_m
    assert result.max_speed_m_s <= simulation.speed_limit_m_s
    assert result.max_tilt_deg <= simulation.tilt_limit_deg
    assert result.max_los_error_deg < 45.0


def test_offline_baseline_is_deterministic() -> None:
    """Identical inputs produce identical metrics for regression testing."""
    first = simulate_interception(default_controller_config())
    second = simulate_interception(default_controller_config())
    assert second == first


def test_frequency_ratio_must_be_integral() -> None:
    """Outer-loop hold steps must map exactly onto inner-loop integration."""
    invalid = OfflineSimulationConfig(outer_loop_hz=60, inner_loop_hz=200)
    with pytest.raises(ValueError, match='divisible'):
        simulate_interception(default_controller_config(), invalid)
