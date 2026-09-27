"""Callback-level range-bootstrap checks for the visual coordinator."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from interception_interfaces.msg import VisionFeature
from px4_msgs.msg import VehicleOdometry
import rclpy
from rclpy.time import Time

from ibvs_control.interception_state_machine import InterceptionPhase
from ibvs_control.takeoff_state_machine import FlightPhase
from ibvs_control.visual_acquisition import (
    VisualAcquisitionCommand,
    VisualAcquisitionPhase,
)
from ibvs_control.vision_interception_coordinator import (
    VisionInterceptionCoordinator,
)


BASE_NS = 100_000_000_000


class _Clock:
    def __init__(self) -> None:
        self.nanoseconds = BASE_NS

    def now(self) -> Time:
        return Time(nanoseconds=self.nanoseconds)


@pytest.fixture
def coordinator(monkeypatch):
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        clock = _Clock()
        monkeypatch.setattr(
            VisionInterceptionCoordinator,
            'get_clock',
            lambda self: clock,
        )
        node.state_machine.phase = FlightPhase.HOLD
        yield node, clock
    finally:
        if node is not None:
            node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def _odometry(height_m: float) -> VehicleOdometry:
    message = VehicleOdometry()
    # PX4 position is NED; a positive ENU height has negative NED down.
    message.position = [0.0, 0.0, -height_m]
    message.velocity = [0.0, 0.0, 0.0]
    message.q = [1.0, 0.0, 0.0, 0.0]
    return message


def _feature(clock: _Clock, height_m: float) -> VisionFeature:
    message = VisionFeature()
    message.capture_stamp = clock.now().to_msg()
    message.valid = True
    # With PX4's identity FRD-to-NED attitude, camera +Z looks north and
    # camera +Y looks down. The static target is 10 m ahead of the camera
    # and at ENU height 1.5 m; y_norm therefore changes with vehicle height.
    message.x_norm = 0.0
    message.y_norm = (height_m - 1.5) / 10.0
    return message


def _send_sample(
    node: VisionInterceptionCoordinator,
    clock: _Clock,
    sample_index: int,
    height_m: float,
) -> None:
    clock.nanoseconds = BASE_NS + sample_index * 250_000_000
    node._odometry_callback(_odometry(height_m))
    node._feature_callback(_feature(clock, height_m))


def test_multiview_bootstrap_needs_parallax_and_estimates_ten_meters(
    coordinator,
) -> None:
    node, clock = coordinator
    assert node.range_estimator.config.model == 'static'

    for index, height_m in enumerate(np.linspace(0.0, 0.05, 20)):
        _send_sample(node, clock, index, height_m)
    assert node.range_estimate is not None
    assert not node.range_estimate.accepted
    assert node.range_estimate.reason in (
        'insufficient_parallax', 'insufficient_transverse_baseline'
    )
    assert node._range_bootstrap_message(clock.nanoseconds) is None

    for index, height_m in enumerate(np.linspace(0.2, 3.0, 21), start=20):
        _send_sample(node, clock, index, height_m)
    assert node.range_estimate is not None
    assert node.range_estimate.accepted
    assert node.range_estimate.quality.parallax_deg > (
        node.range_estimator.config.min_parallax_deg
    )
    assert node._range_bootstrap_message(clock.nanoseconds) is None
    assert node.range_block_reason == 'observability_settling'
    for index in range(41, 53):
        _send_sample(node, clock, index, 3.0)
    bootstrap = node._range_bootstrap_message(clock.nanoseconds)
    assert bootstrap is not None
    # The camera is 0.12 m ahead of the vehicle origin, so a target 10 m
    # ahead of the camera is 10.12 m from the vehicle in the north direction.
    expected_range_m = math.hypot(10.12, 1.5)
    assert bootstrap.range_m == pytest.approx(expected_range_m, abs=0.1)
    assert (bootstrap.p_r.x, bootstrap.p_r.y, bootstrap.p_r.z) == (
        pytest.approx((0.0, -10.12, 1.5), abs=0.1)
    )
    assert bootstrap.sample_count >= node.range_estimator.config.min_samples
    covariance = np.asarray(bootstrap.relative_covariance).reshape((6, 6))
    assert covariance == pytest.approx(covariance.T, abs=1e-10)
    assert np.linalg.eigvalsh(covariance)[0] >= -1e-10
    assert np.all(np.diag(covariance[:3, :3]) >= 0.25)
    assert np.all(np.diag(covariance[3:, 3:]) >= 25.0)
    assert 0.0 < bootstrap.range_std_m <= (
        node.range_estimator.config.max_range_std_m
    )
    assert bootstrap.range_std_m / bootstrap.range_m <= (
        node.bootstrap_max_relative_std
    )


def test_feature_without_capture_time_aligned_pose_is_ignored(
    coordinator,
) -> None:
    node, clock = coordinator
    _send_sample(node, clock, 0, 0.0)
    assert len(node.range_estimator._samples) == 1

    mismatch = _feature(clock, 0.0)
    mismatch.capture_stamp = Time(
        nanoseconds=clock.nanoseconds - 40_000_000
    ).to_msg()
    node._feature_callback(mismatch)
    assert len(node.range_estimator._samples) == 1

    expired = _feature(clock, 0.0)
    expired.capture_stamp = Time(
        nanoseconds=clock.nanoseconds - node.bootstrap_pose_history_ns - 1
    ).to_msg()
    node._feature_callback(expired)
    assert len(node.range_estimator._samples) == 1


def test_ready_gate_requires_bootstrap_sent(coordinator, monkeypatch) -> None:
    node, clock = coordinator
    node.gate.phase = InterceptionPhase.WAIT_HOVER
    node.last_gate_phase = InterceptionPhase.WAIT_HOVER
    node.acquisition_command = VisualAcquisitionCommand(
        phase=VisualAcquisitionPhase.READY,
        yaw_setpoint_rad=0.0,
        vertical_offset_m=0.0,
        image_error=0.0,
        ready=True,
    )
    node.feature = _feature(clock, 1.5)
    node.feature_received_ns = clock.nanoseconds
    node.observer_received_ns = clock.nanoseconds
    node.odometry_received_ns = clock.nanoseconds
    node.command_computed_ns = clock.nanoseconds
    node.state_machine.landed = False
    node.state_machine.armed = True
    node.state_machine.offboard = True
    node.vehicle_speed_m_s = 0.0
    node.px4_command = object()
    node.visual_result = SimpleNamespace(z1=0.0)
    monkeypatch.setattr(node.state_machine, 'step', lambda now_s: ())
    monkeypatch.setattr(node, '_update_visual_acquisition', lambda *args: None)
    monkeypatch.setattr(node, '_publish_visual_acquisition_setpoint',
                        lambda: None)
    monkeypatch.setattr(node, '_publish_rate_mode', lambda: None)
    monkeypatch.setattr(node, '_publish_hover_rate_setpoint', lambda: None)

    node._run_enabled_step(clock.nanoseconds * 1e-9, clock.nanoseconds)
    assert node.gate.phase == InterceptionPhase.WAIT_HOVER

    node.bootstrap_sent = True
    node._run_enabled_step(clock.nanoseconds * 1e-9, clock.nanoseconds)
    assert node.gate.phase == InterceptionPhase.PRESTREAM


def test_range_probe_is_bounded_and_stops_when_lock_is_revoked(
    coordinator,
) -> None:
    node, clock = coordinator
    node.vehicle_heading_ned_rad = 0.0
    for offset_ns in (0, 100_000_000, 700_000_000):
        clock.nanoseconds = BASE_NS + offset_ns
        node._feature_callback(_feature(clock, 1.5))
        node._update_visual_acquisition(
            clock.nanoseconds * 1e-9, clock.nanoseconds
        )
    assert node.acquisition_command.ready
    assert node.range_estimate is None
    assert node.range_probe_started_s is not None

    start_ns = clock.nanoseconds
    period_ns = int(node.range_probe_period_s * 1e9)
    clock.nanoseconds = start_ns + period_ns // 4
    assert node._range_probe_offset_m() == pytest.approx(
        node.range_probe_amplitude_m
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        clock.nanoseconds = start_ns + int(period_ns * fraction)
        assert abs(node._range_probe_offset_m()) <= (
            node.range_probe_amplitude_m + 1e-12
        )

    node.bootstrap_sent = True
    assert node._range_probe_offset_m() == 0.0
    node.bootstrap_sent = False
    lost = VisionFeature()
    lost.valid = False
    node._feature_callback(lost)
    node._update_visual_acquisition(
        clock.nanoseconds * 1e-9, clock.nanoseconds
    )
    assert not node.acquisition_command.ready
    assert node.range_probe_started_s is None
    assert node._range_probe_offset_m() == 0.0
