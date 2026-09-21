"""ROS-message integration for observer-to-paper-controller data flow."""

import numpy as np
import pytest
import rclpy

from interception_interfaces.msg import ObserverState, VisionFeature
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Bool
from ibvs_control.interception_state_machine import InterceptionPhase
from ibvs_control.so3_controller import compute_outer_loop
from ibvs_control.vision_interception_coordinator import (
    VisionInterceptionCoordinator,
)


def test_observer_state_and_onboard_attitude_drive_paper_controller() -> None:
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        state = ObserverState()
        state.initialized = True
        state.valid = True
        state.q = [1.0, 0.0, 0.0, 0.0]
        state.p_r.x = -12.0
        state.p_r.y = 0.0
        state.p_r.z = -1.0
        state.v_r.x = 0.0
        state.v_r.y = 0.0
        state.v_r.z = 0.0
        state.image_xy = [0.0, -1.0 / 12.0]
        node._observer_callback(state)
        odometry = VehicleOdometry()
        odometry.q = [1.0, 0.0, 0.0, 0.0]
        node._odometry_callback(odometry)
        node._update_control(node.get_clock().now().nanoseconds)

        assert node.visual_result is not None
        assert node.inner_result is not None
        assert node.px4_command is not None
        assert node.control_reason == 'ok_paper_observer_control'
        assert np.all(np.isfinite(node.inner_result.omega_d_b))
        assert (
            0.0
            < node.inner_result.thrust_n
            <= node.paper_config.thrust_max_n
        )
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_static_baseline_uses_camera_centre_designed_los() -> None:
    """The static target is commanded along the camera optical axis."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        assert node.designed_los_b == pytest.approx((1.0, 0.0, 0.0))
        assert node.designed_image_xy == pytest.approx((0.0, 0.0))
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_default_outer_loop_does_not_clip_paper_acceleration() -> None:
    """Equation (19) must reach the force calculation without scaling."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        result = compute_outer_loop(
            p_r_e=(-7.0, 0.0, 0.0),
            v_r_e=(2.2, 0.0, 0.0),
            los_e=(1.0, 0.0, 0.0),
            designed_los_e=(1.0, 0.0, 0.0),
            attitude_b_to_e=np.eye(3),
            config=node.paper_config,
        )
        z2_x = 2.2 + node.paper_config.k1 * -7.0
        expected_x = (
            -node.paper_config.k1 * 2.2
            - node.paper_config.k2 * z2_x
            - -7.0
        )
        assert result.acceleration_d_e == pytest.approx(
            (expected_x, 0.0, 0.0)
        )
        assert np.linalg.norm(result.acceleration_d_e) > 0.5
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_nonpaper_p0_paths_are_absent() -> None:
    """Removed P0 branches cannot be re-enabled through node attributes."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        removed = (
            'static_target_mode',
            'enable_acceleration_limits',
            'enable_force_tilt_limit',
            'enable_tilt_recovery',
            'minimum_interception_thrust_n',
            'rate_filter_time_constant_s',
            'thrust_filter_time_constant_s',
        )
        assert all(not hasattr(node, name) for name in removed)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_observer_cannot_command_without_onboard_attitude() -> None:
    """Algorithm 1 requires a measured current R_b^e for attitude feedback."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        state = ObserverState()
        state.initialized = True
        state.valid = True
        state.q = [1.0, 0.0, 0.0, 0.0]
        state.p_r.x = -12.0
        state.image_xy = [0.0, 0.0]
        node._observer_callback(state)
        node._update_control(node.get_clock().now().nanoseconds)

        assert node.visual_result is None
        assert node.inner_result is None
        assert node.px4_command is None
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_px4_odometry_cannot_replace_observer_target_state() -> None:
    """Own attitude is required but cannot replace observer p_r and v_r."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        feature = VisionFeature()
        feature.valid = True
        feature.x_norm = 0.0
        feature.y_norm = 0.0
        feature.area_px = 0.01 * node.image_area_px
        odometry = VehicleOdometry()
        odometry.q = [1.0, 0.0, 0.0, 0.0]
        odometry.velocity = [0.0, 0.0, 0.0]
        odometry.angular_velocity = [0.0, 0.0, 0.0]
        node._feature_callback(feature)
        node._odometry_callback(odometry)
        node._update_control(node.get_clock().now().nanoseconds)

        assert node.visual_result is None
        assert node.inner_result is None
        assert node.px4_command is None

        state = ObserverState()
        state.initialized = True
        state.valid = True
        state.q = [1.0, 0.0, 0.0, 0.0]
        state.p_r.x = -12.0
        state.p_r.y = 0.0
        state.p_r.z = -1.0
        state.v_r.x = -0.5
        state.image_xy = [0.0, -1.0 / 12.0]
        node._observer_callback(state)
        node._update_control(node.get_clock().now().nanoseconds)

        assert node.visual_result is not None
        assert node.inner_result is not None
        assert node.px4_command is not None
        assert node.control_reason == 'ok_paper_observer_control'
        assert np.all(np.isfinite(node.inner_result.omega_d_b))
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_only_verified_contact_commits_the_impact_transition() -> None:
    """Image loss or area cannot replace Gazebo contact plus green feedback."""
    rclpy.init()
    node = None
    try:
        node = VisionInterceptionCoordinator()
        node.gate.phase = InterceptionPhase.ACTIVE
        node._update_visual_terminal(1.0)
        assert not node.visual_interception_complete
        assert node.terminal_started_s is None

        confirmation = Bool()
        confirmation.data = True
        node._contact_callback(confirmation)
        node._update_visual_terminal(1.1)
        assert node.visual_interception_complete
        assert node.terminal_started_s == 1.1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
