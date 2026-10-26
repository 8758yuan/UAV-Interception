"""Tests for the paper moving target confined to the Gazebo plant."""

import math

import pytest

from ibvs_sim.paper_moving_target import (
    PaperTargetTrajectory,
    _pose_request,
)


@pytest.mark.parametrize('pattern', ('figure8', 'circle'))
def test_trajectory_starts_at_spawn_pose_and_holds_requested_speed(pattern) -> None:
    trajectory = PaperTargetTrajectory(
        pattern=pattern,
        origin=(12.0, 0.0, 3.0),
        speed_m_s=5.0,
        radius_x_m=4.0,
        radius_y_m=2.0 if pattern == 'figure8' else 4.0,
    )
    start = trajectory.sample(0.0)
    assert start.position == pytest.approx((12.0, 0.0, 3.0))
    for time_s in (0.1, 0.7, 1.9, 4.2):
        sample = trajectory.sample(time_s)
        assert math.hypot(*sample.velocity[:2]) == pytest.approx(5.0)
        assert sample.velocity[2] == 0.0


def test_figure8_is_periodic_and_crosses_its_origin() -> None:
    trajectory = PaperTargetTrajectory(
        pattern='figure8',
        origin=(12.0, 0.0, 3.0),
        speed_m_s=7.5,
        radius_x_m=4.0,
        radius_y_m=2.0,
    )
    period_s = trajectory.path_length_m / trajectory.speed_m_s
    assert trajectory.sample(period_s).position == pytest.approx(
        trajectory.sample(0.0).position,
        abs=1e-8,
    )
    assert trajectory.sample(0.5 * period_s).position == pytest.approx(
        (12.0, 0.0, 3.0),
        abs=1e-6,
    )


def test_pose_request_contains_no_velocity_or_ros_truth_interface() -> None:
    trajectory = PaperTargetTrajectory(
        pattern='figure8',
        origin=(12.0, 0.0, 3.0),
        speed_m_s=10.0,
        radius_x_m=4.0,
        radius_y_m=2.0,
    )
    sample = trajectory.sample(0.25)
    request = _pose_request('ibvs_target', sample)
    assert request.name == 'ibvs_target'
    assert request.position.x == pytest.approx(sample.position[0])
    assert request.position.z == pytest.approx(3.0)
    assert request.orientation.w**2 + request.orientation.z**2 == pytest.approx(1.0)


def test_invalid_moving_target_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        PaperTargetTrajectory(
            pattern='random',
            origin=(12.0, 0.0, 3.0),
            speed_m_s=5.0,
            radius_x_m=4.0,
            radius_y_m=2.0,
        )


def test_tilted_circle_radius_speed_and_direction() -> None:
    trajectory = PaperTargetTrajectory(
        pattern='circle', origin=(12.0, 0.0, 4.0), speed_m_s=5.0,
        radius_x_m=6.0, radius_y_m=6.0, tilt_deg=45.0,
    )
    offset = 6.0 / math.sqrt(2.0)
    centre = (12.0 - offset, 0.0, 4.0 - offset)
    period = trajectory.path_length_m / trajectory.speed_m_s
    for fraction in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        sample = trajectory.sample(fraction * period)
        x, _, z = sample.position
        assert math.dist(sample.position, centre) == pytest.approx(6.0)
        assert z - 4.0 == pytest.approx(x - 12.0)
        assert math.hypot(*sample.velocity) == pytest.approx(5.0)
    assert trajectory.sample(0.0).position == pytest.approx((12.0, 0.0, 4.0))
    assert trajectory.sample(period).position == pytest.approx((12.0, 0.0, 4.0))
    assert trajectory.sample(0.5 * period).position[2] == pytest.approx(
        4.0 - 2.0 * offset,
    )
