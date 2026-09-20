"""End-to-end static-target verification through the 18-state observer."""

import math

from ibvs_control.observer_closed_loop import simulate_observer_closed_loop
from ibvs_control.offline_interception import default_controller_config


def test_no_delay_observer_controller_closes_static_target_loop() -> None:
    result = simulate_observer_closed_loop(default_controller_config())
    assert result.hit
    assert result.safe
    assert result.reason == 'hit'
    assert result.maximum_speed_m_s < 2.0
    assert result.maximum_tilt_deg < 20.0
    assert result.maximum_image_radius < math.tan(math.radians(39.0))
    assert result.prediction_count > 6000
    assert result.correction_count > 900
    assert result.maximum_velocity_error_m_s < 0.25
    assert result.target_distance_m == 0.0


def test_no_delay_observer_controller_closes_moving_target_loop() -> None:
    """Target velocity exists only in the plant, never in the control path."""
    result = simulate_observer_closed_loop(
        default_controller_config(),
        target_velocity_e=(0.0, 0.15, 0.0),
    )
    assert result.hit
    assert result.safe
    assert result.reason == 'hit'
    assert result.target_distance_m > 4.0
    assert result.maximum_image_radius < math.tan(math.radians(39.0))
    assert result.maximum_velocity_error_m_s < 0.35
