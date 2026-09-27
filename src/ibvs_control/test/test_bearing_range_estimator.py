"""Tests for multi-view bearing range estimation without ROS or simulation."""

from dataclasses import replace

import numpy as np
import pytest

from ibvs_control.bearing_range_estimator import (
    BearingRangeConfig,
    BearingRangeEstimator,
    BearingSample,
)


def _sample(time_s, camera, target, rotation=None):
    if rotation is None:
        rotation = np.eye(3)
    relative_camera = rotation.T @ (
        np.asarray(target) - np.asarray(camera)
    )
    image = relative_camera[:2] / relative_camera[2]
    return BearingSample(
        timestamp_s=time_s,
        image_xy=tuple(image),
        camera_position_enu_m=tuple(camera),
        camera_to_enu_rotation=rotation,
    )


def test_static_target_triangulation_returns_metric_position_and_covariance():
    target = np.array((1.4, -0.6, 13.0))
    estimator = BearingRangeEstimator()
    for index in range(12):
        camera = (0.12 * index, 0.15 * np.sin(index / 3.0), 0.0)
        assert estimator.add(_sample(index * 0.1, camera, target))

    result = estimator.estimate()
    assert result.accepted, result.reason
    assert result.reference_time_s == pytest.approx(1.1)
    assert result.target_position_enu_m == pytest.approx(target, abs=1e-8)
    assert result.target_velocity_enu_m_s is None
    assert result.covariance_enu.shape == (3, 3)
    assert np.linalg.eigvalsh(result.covariance_enu)[0] > 0.0
    assert result.quality.parallax_deg > 0.5
    assert result.quality.transverse_baseline_m > 1.0
    assert result.quality.reprojection_rms < 1e-10
    assert result.quality.range_std_m < 5.0


def test_rotating_without_translation_cannot_estimate_range():
    target = np.array((0.0, 0.0, 12.0))
    estimator = BearingRangeEstimator()
    for index, angle in enumerate(np.linspace(-0.2, 0.2, 8)):
        rotation = np.array((
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        ))
        estimator.add(_sample(index * 0.1, (0.0, 0.0, 0.0),
                              target, rotation))

    result = estimator.estimate()
    assert not result.accepted
    assert result.reason == 'insufficient_parallax'
    assert result.target_position_enu_m is None


def test_axial_translation_without_parallax_cannot_estimate_range():
    estimator = BearingRangeEstimator()
    for index in range(8):
        estimator.add(_sample(index * 0.1, (0.0, 0.0, index * 0.1),
                              (0.0, 0.0, 12.0)))
    result = estimator.estimate()
    assert not result.accepted
    assert result.reason == 'insufficient_parallax'


def test_constant_velocity_target_needs_nonuniform_camera_motion():
    config = BearingRangeConfig(
        model='constant_velocity', max_range_std_m=10.0,
        max_velocity_std_m_s=5.0,
    )
    estimator = BearingRangeEstimator(config)
    position_at_latest = np.array((2.5, -0.8, 13.0))
    velocity = np.array((0.5, 0.15, -0.1))
    times = np.linspace(0.0, 3.0, 20)
    for time_s in times:
        camera = (0.2 * time_s + 0.13 * time_s ** 2,
                  0.12 * time_s ** 2, 0.04 * time_s)
        target = position_at_latest + velocity * (time_s - times[-1])
        estimator.add(_sample(time_s, camera, target))

    result = estimator.estimate()
    assert result.accepted, (result.reason, result.quality)
    assert result.reference_time_s == pytest.approx(times[-1])
    assert result.target_position_enu_m == pytest.approx(
        position_at_latest, abs=1e-7
    )
    assert result.target_velocity_enu_m_s == pytest.approx(
        velocity, abs=1e-7
    )
    assert result.covariance_enu.shape == (6, 6)
    assert np.linalg.eigvalsh(result.covariance_enu)[0] > 0.0
    assert result.quality.velocity_std_m_s < 5.0


def test_constant_speed_camera_motion_rejects_dynamic_scale_ambiguity():
    estimator = BearingRangeEstimator(BearingRangeConfig(
        model='constant_velocity',
    ))
    for time_s in np.linspace(0.0, 2.0, 16):
        camera = np.array((0.4 * time_s, 0.1 * time_s, 0.0))
        target = np.array((1.0 + 0.15 * time_s,
                           0.2 - 0.1 * time_s, 12.0))
        estimator.add(_sample(time_s, camera, target))

    result = estimator.estimate()
    assert not result.accepted
    assert result.reason in (
        'unobservable_geometry', 'ill_conditioned_geometry'
    )


def test_single_large_image_outlier_is_removed_before_covariance():
    estimator = BearingRangeEstimator()
    target = (1.0, 0.3, 12.0)
    for index in range(12):
        sample = _sample(index * 0.1,
                         (0.13 * index, 0.05 * index, 0.0), target)
        if index == 5:
            sample = replace(sample, image_xy=(
                sample.image_xy[0] + 0.2, sample.image_xy[1] - 0.2
            ))
        estimator.add(sample)

    result = estimator.estimate()
    assert result.accepted, result.reason
    assert result.target_position_enu_m == pytest.approx(target, abs=1e-6)
    assert result.quality.inlier_count == 11
    assert result.quality.outlier_count == 1
    assert result.quality.max_reprojection_error > 0.2
    assert result.quality.reprojection_rms < 1e-8


def test_many_large_image_outliers_reject_window():
    estimator = BearingRangeEstimator(BearingRangeConfig(
        min_samples=6, max_samples=25, max_age_s=5.0,
    ))
    target = (1.0, 0.3, 12.0)
    for index in range(20):
        sample = _sample(index * 0.1,
                         (0.12 * index, 0.03 * index, 0.0), target)
        if index % 3 == 0:
            sample = replace(sample, image_xy=(
                sample.image_xy[0] + 0.25,
                sample.image_xy[1] - 0.2,
            ))
        estimator.add(sample)

    result = estimator.estimate()
    assert not result.accepted
    assert result.reason == 'excess_outlier_fraction'
    assert result.quality.outlier_count > 4


def test_single_outlier_in_long_window_cannot_hide_below_rms_limit():
    estimator = BearingRangeEstimator(BearingRangeConfig(
        max_samples=200, max_age_s=30.0,
    ))
    target = (1.0, 0.3, 12.0)
    for index in range(200):
        sample = _sample(index * 0.1,
                         (0.02 * index, 0.003 * index, 0.0), target)
        if index == 100:
            sample = replace(sample, image_xy=(
                sample.image_xy[0] + 0.2,
                sample.image_xy[1] - 0.2,
            ))
        estimator.add(sample)

    result = estimator.estimate()
    assert result.accepted, result.reason
    assert result.quality.outlier_count == 1
    assert result.quality.max_reprojection_error > 0.2
    assert result.target_position_enu_m == pytest.approx(target, abs=1e-6)


def test_window_discards_old_samples_and_stale_timestamps():
    estimator = BearingRangeEstimator(BearingRangeConfig(
        min_samples=3, max_samples=4, max_age_s=0.25,
    ))
    target = (0.0, 0.0, 10.0)
    for index in range(6):
        estimator.add(_sample(index * 0.1, (0.1 * index, 0.0, 0.0),
                              target))
    assert not estimator.add(_sample(0.4, (0.4, 0.0, 0.0), target))
    assert estimator.estimate().quality.sample_count == 3
    estimator.clear()
    assert estimator.estimate().reason == 'insufficient_samples'


def test_invalid_pose_and_configuration_are_rejected():
    with pytest.raises(ValueError, match='orthonormal'):
        BearingRangeEstimator().add(BearingSample(
            0.0, (0.0, 0.0), (0.0, 0.0, 0.0), np.zeros((3, 3))
        ))
    with pytest.raises(ValueError, match='min_samples'):
        BearingRangeEstimator(BearingRangeConfig(min_samples=2))
