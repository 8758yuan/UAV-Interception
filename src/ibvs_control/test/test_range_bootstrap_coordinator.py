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
    # and at ENU height 1.5 m. The camera and body origin share a height.
    message.width_norm = 0.26 / 10.0
    message.x_norm = 0.0
    message.y_norm = (height_m - 1.5) / 10.0
    return message


def _send_frame(node, clock, height_m=3.0):
    node._odometry_callback(_odometry(height_m))
    node._feature_callback(_feature(clock, height_m))


@pytest.mark.parametrize('relative_altitude_m', [-1.0, 60.0])
def test_visual_search_publishes_height_without_fence(
    coordinator, relative_altitude_m,
):
    node, _ = coordinator
    node.state_machine.home_x = 1.0
    node.state_machine.home_y = 2.0
    node.state_machine.home_z = 3.0
    node.acquisition_command = SimpleNamespace(
        vertical_offset_m=(
            node.state_machine.config.target_altitude_m - relative_altitude_m
        ),
        yaw_setpoint_rad=0.4,
    )
    modes, setpoints = [], []
    node.offboard_pub = SimpleNamespace(publish=modes.append)
    node.trajectory_pub = SimpleNamespace(publish=setpoints.append)

    node._publish_visual_acquisition_setpoint()

    assert modes[0].position
    assert list(setpoints[0].position) == pytest.approx(
        [1.0, 2.0, 3.0 - relative_altitude_m]
    )
    assert setpoints[0].yaw == pytest.approx(0.4)


def test_one_frame_initializes_without_parallax_or_settling(coordinator):
    node, clock = coordinator
    _send_frame(node, clock)
    bootstrap = node._range_bootstrap_message(clock.nanoseconds)
    assert bootstrap is not None
    # Camera is 0.14 m forward; optical depth is 10 m, not radial distance.
    assert (bootstrap.p_r.x, bootstrap.p_r.y, bootstrap.p_r.z) == (
        pytest.approx((0.0, -10.14, 1.5))
    )
    assert bootstrap.range_m == pytest.approx(math.hypot(10.14, 1.5))
    assert bootstrap.sample_count == 1
    assert bootstrap.range_std_m > 2.0  # Coarse distance is accepted.
    covariance = np.asarray(bootstrap.relative_covariance).reshape((6, 6))
    assert covariance == pytest.approx(covariance.T)
    assert np.linalg.eigvalsh(covariance)[0] > 0.0
    assert np.diag(covariance[3:, 3:]) == pytest.approx((25.0,) * 3)


def test_delayed_image_uses_capture_pose_and_current_own_velocity(coordinator):
    node, clock = coordinator
    _send_frame(node, clock)
    clock.nanoseconds += 80_000_000
    node._odometry_callback(_odometry(3.5))
    node.vehicle_velocity_ned = np.array((1.0, 2.0, -0.3))
    bootstrap = node._range_bootstrap_message(clock.nanoseconds)
    assert bootstrap is not None
    assert (bootstrap.p_r.x, bootstrap.p_r.y, bootstrap.p_r.z) == (
        pytest.approx((0.0, -10.14, 2.0))
    )
    assert (bootstrap.v_r.x, bootstrap.v_r.y, bootstrap.v_r.z) == (
        pytest.approx((2.0, 1.0, 0.3))
    )
    covariance = np.asarray(bootstrap.relative_covariance).reshape((6, 6))
    assert covariance[:3, 3:] == pytest.approx(0.08 * 25.0 * np.eye(3))


@pytest.mark.parametrize('width', (0.0, -0.01, math.nan, math.inf))
def test_invalid_box_width_cannot_initialize(coordinator, width):
    node, clock = coordinator
    _send_frame(node, clock)
    node.feature.width_norm = width
    assert node._range_bootstrap_message(clock.nanoseconds) is None
    assert node.range_block_reason == 'invalid_size_measurement'


def test_capture_pose_mismatch_and_stale_image_cannot_initialize(coordinator):
    node, clock = coordinator
    _send_frame(node, clock)
    node.feature.capture_stamp = Time(
        nanoseconds=clock.nanoseconds - 40_000_000
    ).to_msg()
    assert node._range_bootstrap_message(clock.nanoseconds) is None
    assert node.range_block_reason == 'waiting_for_capture_pose'
    node.feature.capture_stamp = Time(
        nanoseconds=clock.nanoseconds - 600_000_000
    ).to_msg()
    assert node._range_bootstrap_message(clock.nanoseconds) is None
    assert node.range_block_reason == 'size_measurement_stale'


def test_ready_gate_requires_bootstrap_sent(coordinator, monkeypatch) -> None:
    node, clock = coordinator
    node.gate.phase = InterceptionPhase.WAIT_HOVER
    node.last_gate_phase = InterceptionPhase.WAIT_HOVER
    node.vehicle_heading_ned_rad = 0.0
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
    monkeypatch.setattr(node, '_publish_visual_acquisition_setpoint',
                        lambda: None)
    monkeypatch.setattr(node, '_publish_rate_mode', lambda: None)
    monkeypatch.setattr(node, '_publish_hover_rate_setpoint', lambda: None)

    node._run_enabled_step(clock.nanoseconds * 1e-9, clock.nanoseconds)
    assert node.acquisition_command.phase == VisualAcquisitionPhase.READY
    assert node.acquisition_command.ready
    assert node.gate.phase == InterceptionPhase.WAIT_HOVER

    node.bootstrap_sent = True
    node._run_enabled_step(clock.nanoseconds * 1e-9, clock.nanoseconds)
    assert node.gate.phase == InterceptionPhase.PRESTREAM


def test_first_ready_frame_sends_one_bootstrap(coordinator, monkeypatch):
    node, clock = coordinator
    node.vehicle_heading_ned_rad = 0.0
    _send_frame(node, clock)
    published = []
    resets = []
    monkeypatch.setattr(
        node.observer_bootstrap_pub, 'publish', published.append
    )
    monkeypatch.setattr(node.observer_reset_pub, 'publish', resets.append)
    now_s = clock.nanoseconds * 1e-9
    node._update_visual_acquisition(now_s, clock.nanoseconds)
    assert node.acquisition_command.ready
    assert node.bootstrap_sent
    assert len(published) == len(resets) == 1
    assert published[0].sample_count == 1
    node._update_visual_acquisition(now_s, clock.nanoseconds)
    assert len(published) == len(resets) == 1
