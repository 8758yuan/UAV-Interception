"""Tests for controller-shadow metric accumulation."""

import pytest

from ibvs_control.shadow_metrics import ShadowMetrics


def test_summary_separates_validity_and_saturations() -> None:
    """Only valid samples enter controller saturation denominators."""
    metrics = ShadowMetrics(k_b=0.3)
    metrics.update(stamp_s=10.0, valid=False, reason='waiting')
    metrics.update(
        stamp_s=10.005,
        valid=True,
        reason='ok',
        z1=0.1,
        thrust_n=20.0,
        thrust_normalized=0.7,
        rate_saturated=True,
    )
    metrics.update(
        stamp_s=10.010,
        valid=True,
        reason='ok',
        z1=0.2,
        thrust_n=26.9,
        thrust_normalized=1.0,
        thrust_saturated=True,
        thrust_mapping_saturated=True,
    )

    summary = metrics.summary()
    assert summary.sample_count == 3
    assert summary.valid_count == 2
    assert summary.duration_s == pytest.approx(0.01)
    assert summary.sample_rate_hz == pytest.approx(200.0)
    assert summary.valid_fraction == pytest.approx(2.0 / 3.0)
    assert summary.min_barrier_margin == pytest.approx(0.1)
    assert summary.max_abs_z1 == pytest.approx(0.2)
    assert summary.max_thrust_n == pytest.approx(26.9)
    assert summary.max_thrust_normalized == 1.0
    assert summary.thrust_saturation_fraction == 0.5
    assert summary.thrust_mapping_saturation_fraction == 0.5
    assert summary.rate_saturation_fraction == 0.5
    assert summary.invalid_reasons == {'waiting': 1}


def test_empty_summary_is_well_defined() -> None:
    """An empty capture produces finite zero fractions."""
    summary = ShadowMetrics(k_b=0.3).summary()
    assert summary.sample_count == 0
    assert summary.valid_fraction == 0.0
    assert summary.rate_saturation_fraction == 0.0
    assert summary.min_barrier_margin is None


def test_nonmonotonic_timestamps_are_rejected() -> None:
    """A ROS time reset cannot silently corrupt an observation."""
    metrics = ShadowMetrics(k_b=0.3)
    metrics.update(stamp_s=2.0, valid=False, reason='waiting')
    with pytest.raises(ValueError, match='monotonic'):
        metrics.update(stamp_s=1.0, valid=False, reason='waiting')
