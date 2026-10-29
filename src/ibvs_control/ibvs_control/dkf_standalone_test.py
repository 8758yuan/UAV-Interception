"""Run independent DKF trials using PX4 position control only."""

from dataclasses import replace
import math

from interception_interfaces.msg import ControlDebug
from px4_msgs.msg import TrajectorySetpoint
import rclpy
from rclpy.executors import ExternalShutdownException

from ibvs_control.takeoff_state_machine import FlightPhase
from ibvs_control.vision_interception_coordinator import (
    VisionInterceptionCoordinator,
)


def position_offset_enu(test_case, elapsed_s):
    """Return ENU offsets: B sinusoids, C figure eight, D circle, A hover."""
    if test_case == 'B':
        return (
            4.0 * math.sin(2.0 * math.pi * elapsed_s / 20.0),
            6.0 * math.sin(2.0 * math.pi * elapsed_s / 15.0),
            2.0 * math.sin(2.0 * math.pi * elapsed_s / 18.0),
        )
    if test_case == 'C':
        theta = 2.0 * math.pi * elapsed_s / 20.0
        return (
            4.0 * math.sin(theta),
            6.0 * math.sin(2.0 * theta),
            2.0 * math.sin(theta),
        )
    if test_case == 'D':
        # Reflect C's target circle in ENU X so the vehicle moves forward.
        theta = elapsed_s / 3.0
        up = 6.0 * (math.cos(theta) - 1.0) / math.sqrt(2.0)
        return (-up, 6.0 * math.sin(theta), up)
    return (0.0, 0.0, 0.0)


class DkfStandaloneTest(VisionInterceptionCoordinator):
    """Reuse takeoff and size bootstrap, bypassing the interception loop."""

    def __init__(self):
        """Configure the selected trial and remove the inherited rate output."""
        super().__init__()
        self.declare_parameter('test_case', 'C')
        self.test_case = str(self.get_parameter('test_case').value)
        if self.test_case not in ('A', 'B', 'C', 'D'):
            raise ValueError('test_case must be A, B, C or D')
        self.duration_s = 20.0 if self.test_case == 'A' else 60.0
        self.test_start_ns = None
        # The inherited initialization creates this publisher when enabled;
        # this test has no rate/thrust command path.
        if self.rates_pub is not None:
            self.destroy_publisher(self.rates_pub)
            self.rates_pub = None
        self.state_machine.config = replace(
            self.state_machine.config, auto_land=False,
        )
        self.get_logger().info(
            f'DKF 独立测试 {self.test_case}：{self.duration_s:.0f} s，'
            '仅使用 PX4 位置控制；等待起飞和尺寸初始化'
        )

    def timer_callback(self):
        """Run the existing takeoff lifecycle and an independent test window."""
        if not self.command_output_enabled:
            return
        now_ns = self.get_clock().now().nanoseconds
        now_s = now_ns * 1e-9
        actions = self.state_machine.step(now_s)
        self._log_phase_change()
        if (self.state_machine.phase == FlightPhase.HOLD
                and self.test_start_ns is None):
            bootstrap = self._range_bootstrap_message(now_ns)
            if bootstrap is not None:
                self._reset_observer_for_interception(bootstrap)
                self.test_start_ns = now_ns
                # Keep the existing timed HOLD -> LAND transition, with its
                # clock origin aligned to the actual test start.
                self.state_machine.config = replace(
                    self.state_machine.config, auto_land=True,
                    hold_duration_s=(
                        now_s - self.state_machine.phase_entered_s
                        + self.duration_s
                    ),
                )
                self.get_logger().info(f'DKF 独立测试 {self.test_case} 开始')
            elif self.range_block_reason != self.last_range_reason:
                self.last_range_reason = self.range_block_reason
                self.get_logger().info(
                    f'等待尺寸初始化：{self.range_block_reason}'
                )
        self._execute_actions(actions)
        status = ControlDebug()
        status.stamp = self.get_clock().now().to_msg()
        status.control_state = (
            'ACTIVE' if self.test_start_ns is not None
            and self.state_machine.phase == FlightPhase.HOLD
            else self.state_machine.phase.value
        )
        status.reason = self.state_machine.abort_reason or f'DKF test {self.test_case}'
        self.debug_pub.publish(status)
        self.finished = self.state_machine.phase == FlightPhase.COMPLETE or (
            self.state_machine.phase == FlightPhase.ABORT
            and not self.state_machine.armed and self.state_machine.landed
        )

    def publish_trajectory_setpoint(self):
        """Publish position-only test motion in NED, retaining the final point."""
        if self.test_start_ns is None:
            super().publish_trajectory_setpoint()
            return
        elapsed_s = min(
            (self.get_clock().now().nanoseconds - self.test_start_ns) * 1e-9,
            self.duration_s,
        )
        east, north, up = position_offset_enu(self.test_case, elapsed_s)
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp_us()
        message.position = [
            self.state_machine.home_x + north,
            self.state_machine.home_y + east,
            self.state_machine.target_z_ned_m - up,
        ]
        message.yaw = self.target_yaw_rad
        self.trajectory_pub.publish(message)


def main(args=None):
    """Finish after landing so launch can finalize the existing DKF reports."""
    rclpy.init(args=args)
    node = DkfStandaloneTest()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
