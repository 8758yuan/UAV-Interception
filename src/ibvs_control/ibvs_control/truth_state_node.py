"""Publish P2 truth relative state from PX4 and Gazebo odometry."""

import math
from typing import Optional

from interception_interfaces.msg import RelativeState
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_control.frames import ned_to_enu
from ibvs_control.relative_state import compute_relative_kinematics


class TruthStateNode(Node):
    """Combine interceptor NED odometry with target ENU truth odometry."""

    def __init__(self) -> None:
        super().__init__('truth_state')
        self.declare_parameter(
            'vehicle_odometry_topic',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter(
            'target_odometry_topic',
            '/model/ibvs_target/odometry',
        )
        self.declare_parameter(
            'relative_state_topic',
            '/interception/truth/relative_state',
        )
        self.declare_parameter('target_timeout_s', 0.5)

        self.target_timeout_s = float(
            self.get_parameter('target_timeout_s').value
        )
        if not math.isfinite(self.target_timeout_s) or self.target_timeout_s <= 0:
            raise ValueError('target_timeout_s must be finite and positive')
        self.target_odometry: Optional[Odometry] = None
        self.target_received_ns: Optional[int] = None
        self.warned_target_missing = False
        self.warned_frame = False

        self.publisher = self.create_publisher(
            RelativeState,
            str(self.get_parameter('relative_state_topic').value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('target_odometry_topic').value),
            self._target_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter('vehicle_odometry_topic').value),
            self._vehicle_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'P2-A truth monitor ready; no flight-control commands are sent'
        )

    def _target_callback(self, msg: Odometry) -> None:
        self.target_odometry = msg
        self.target_received_ns = self.get_clock().now().nanoseconds
        self.warned_target_missing = False

    def _vehicle_callback(self, msg: VehicleOdometry) -> None:
        now = self.get_clock().now()
        if self.target_odometry is None or self.target_received_ns is None:
            if not self.warned_target_missing:
                self.get_logger().warning('Waiting for Gazebo target odometry')
                self.warned_target_missing = True
            return
        target_age_s = (now.nanoseconds - self.target_received_ns) * 1e-9
        if target_age_s > self.target_timeout_s:
            if not self.warned_target_missing:
                self.get_logger().warning('Gazebo target odometry timed out')
                self.warned_target_missing = True
            return
        if (
            msg.pose_frame != VehicleOdometry.POSE_FRAME_NED
            or msg.velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED
        ):
            if not self.warned_frame:
                self.get_logger().error(
                    'VehicleOdometry must use NED position and velocity frames'
                )
                self.warned_frame = True
            return

        target = self.target_odometry
        try:
            state = compute_relative_kinematics(
                ned_to_enu(msg.position),
                ned_to_enu(msg.velocity),
                (
                    target.pose.pose.position.x,
                    target.pose.pose.position.y,
                    target.pose.pose.position.z,
                ),
                (
                    target.twist.twist.linear.x,
                    target.twist.twist.linear.y,
                    target.twist.twist.linear.z,
                ),
            )
        except ValueError as error:
            self.get_logger().error(f'Invalid truth state: {error}')
            return

        output = RelativeState()
        output.stamp = now.to_msg()
        _assign_vector(output.p_r, state.p_r)
        _assign_vector(output.v_r, state.v_r)
        _assign_vector(output.los, state.los)
        output.source = 'truth'
        self.publisher.publish(output)


def _assign_vector(message, values) -> None:
    message.x, message.y, message.z = values


def main(args=None) -> None:
    """Run the P2-A truth-state publisher."""
    rclpy.init(args=args)
    node = TruthStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
