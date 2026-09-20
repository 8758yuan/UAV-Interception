"""Opt-in ROS adapter from ControlDebug to PX4 rate setpoints."""

import math
from typing import Optional

from interception_interfaces.msg import ControlDebug
from px4_msgs.msg import VehicleRatesSetpoint
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_control.px4_command_adapter import adapt_rate_thrust_command
from ibvs_control.thrust_mapping import ThrustMappingConfig


CONFIRMATION_TOKEN = 'ENABLE_RATE_THRUST_CONTROL'


class Px4CommandAdapterNode(Node):
    """Publish only when both the Boolean and explicit token opt in."""

    def __init__(self) -> None:
        super().__init__('px4_command_adapter')
        self.declare_parameter('enable_control', False)
        self.declare_parameter('confirmation_token', '')
        self.declare_parameter('mass_kg', 2.0)
        self.declare_parameter('hover_thrust_normalized', 0.727)
        self.declare_parameter('omega_limit_rad_s', 0.5)
        self.declare_parameter('command_timeout_s', 0.1)
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('debug_topic', '/interception/control/debug')
        self.declare_parameter(
            'rates_setpoint_topic',
            '/fmu/in/vehicle_rates_setpoint',
        )

        self.enabled = bool(self.get_parameter('enable_control').value)
        token = str(self.get_parameter('confirmation_token').value)
        if self.enabled and token != CONFIRMATION_TOKEN:
            raise ValueError('enable_control requires the confirmation token')
        self.omega_limit_rad_s = self._positive('omega_limit_rad_s')
        self.command_timeout_s = self._positive('command_timeout_s')
        publish_rate_hz = self._positive('publish_rate_hz')
        self.mapping = ThrustMappingConfig(
            mass_kg=self._positive('mass_kg'),
            hover_thrust_normalized=self._positive(
                'hover_thrust_normalized'
            ),
        )
        self.mapping.validate()
        self.latest: Optional[ControlDebug] = None
        self.received_ns: Optional[int] = None
        self.publisher = None
        if self.enabled:
            self.publisher = self.create_publisher(
                VehicleRatesSetpoint,
                str(self.get_parameter('rates_setpoint_topic').value),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                ControlDebug,
                str(self.get_parameter('debug_topic').value),
                self._callback,
                qos_profile_sensor_data,
            )
            self.create_timer(1.0 / publish_rate_hz, self._timer_callback)
            self.get_logger().warning(
                'PX4 command adapter ENABLED; external state machine '
                'must manage Offboard, arming, and landing'
            )
        else:
            self.get_logger().warning(
                'PX4 command adapter disabled; no PX4 publisher created'
            )

    def _positive(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def _callback(self, message: ControlDebug) -> None:
        self.latest = message
        self.received_ns = self.get_clock().now().nanoseconds

    def _timer_callback(self) -> None:
        if self.latest is None or self.received_ns is None:
            return
        age_s = (
            self.get_clock().now().nanoseconds - self.received_ns
        ) * 1e-9
        if age_s > self.command_timeout_s or not self.latest.valid:
            return
        command = adapt_rate_thrust_command(
            (
                self.latest.omega_d_b.x,
                self.latest.omega_d_b.y,
                self.latest.omega_d_b.z,
            ),
            self.latest.thrust_n,
            self.omega_limit_rad_s,
            self.mapping,
        )
        message = VehicleRatesSetpoint()
        message.timestamp = self.get_clock().now().nanoseconds // 1000
        message.roll, message.pitch, message.yaw = command.rates_frd_rad_s
        message.thrust_body = list(command.thrust_body)
        message.reset_integral = False
        self.publisher.publish(message)


def main(args=None) -> None:
    """Run the opt-in command adapter."""
    rclpy.init(args=args)
    node = Px4CommandAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
