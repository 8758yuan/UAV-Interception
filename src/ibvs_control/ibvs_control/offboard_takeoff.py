"""
Minimal PX4 Offboard position-setpoint takeoff node.

This node is the P0 baseline used to verify that the ROS 2/PX4 transport is
working.  The counter-based mode and arm sequence will be replaced by the P1
state machine after this package baseline has been validated.
"""

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class OffboardTakeoff(Node):
    """Publish a fixed local-position setpoint and request Offboard takeoff."""

    def __init__(self) -> None:
        super().__init__('offboard_takeoff')

        self.declare_parameter(
            'topics.offboard_control_mode',
            '/fmu/in/offboard_control_mode',
        )
        self.declare_parameter(
            'topics.trajectory_setpoint',
            '/fmu/in/trajectory_setpoint',
        )
        self.declare_parameter(
            'topics.vehicle_command',
            '/fmu/in/vehicle_command',
        )
        self.declare_parameter(
            'topics.vehicle_local_position',
            '/fmu/out/vehicle_local_position_v1',
        )
        self.declare_parameter('timer_period_s', 0.1)
        self.declare_parameter('target_altitude_m', 2.0)
        self.declare_parameter('target_yaw_rad', 0.0)
        self.declare_parameter('offboard_request_count', 20)
        self.declare_parameter('arm_request_count', 25)
        self.declare_parameter('target_system', 1)
        self.declare_parameter('target_component', 1)
        self.declare_parameter('source_system', 1)
        self.declare_parameter('source_component', 1)

        self.target_altitude_m = float(
            self.get_parameter('target_altitude_m').value
        )
        self.target_yaw_rad = float(
            self.get_parameter('target_yaw_rad').value
        )
        self.offboard_request_count = int(
            self.get_parameter('offboard_request_count').value
        )
        self.arm_request_count = int(
            self.get_parameter('arm_request_count').value
        )
        self.target_system = int(self.get_parameter('target_system').value)
        self.target_component = int(
            self.get_parameter('target_component').value
        )
        self.source_system = int(self.get_parameter('source_system').value)
        self.source_component = int(
            self.get_parameter('source_component').value
        )

        if self.target_altitude_m <= 0.0:
            raise ValueError('target_altitude_m must be greater than zero')
        if self.offboard_request_count <= 0:
            raise ValueError('offboard_request_count must be greater than zero')
        if self.arm_request_count <= self.offboard_request_count:
            raise ValueError(
                'arm_request_count must be greater than offboard_request_count'
            )

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self._topic('topics.offboard_control_mode'),
            qos_pub,
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            self._topic('topics.trajectory_setpoint'),
            qos_pub,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            self._topic('topics.vehicle_command'),
            qos_pub,
        )
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            self._topic('topics.vehicle_local_position'),
            self.position_callback,
            qos_sub,
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.position_received = False
        self.counter = 0

        timer_period_s = float(self.get_parameter('timer_period_s').value)
        if timer_period_s <= 0.0:
            raise ValueError('timer_period_s must be greater than zero')
        self.timer = self.create_timer(timer_period_s, self.timer_callback)

        self.get_logger().info(
            f'Waiting for local position; target altitude is '
            f'{self.target_altitude_m:.2f} m'
        )

    def _topic(self, parameter_name: str) -> str:
        """Return a validated topic-name parameter."""
        topic = str(self.get_parameter(parameter_name).value)
        if not topic:
            raise ValueError(f'{parameter_name} must not be empty')
        return topic

    def _timestamp_us(self) -> int:
        """Return the ROS clock timestamp in microseconds for PX4 messages."""
        return self.get_clock().now().nanoseconds // 1000

    def position_callback(self, msg: VehicleLocalPosition) -> None:
        """Latch the horizontal position used for the vertical takeoff."""
        if self.position_received:
            return

        self.current_x = float(msg.x)
        self.current_y = float(msg.y)
        self.position_received = True
        self.get_logger().info(
            f'Initial position: x={self.current_x:.2f}, '
            f'y={self.current_y:.2f}'
        )

    def publish_offboard_control_mode(self) -> None:
        """Select position control as the active Offboard command type."""
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self) -> None:
        """Hold the initial horizontal position and climb in PX4 NED."""
        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position = [
            self.current_x,
            self.current_y,
            -self.target_altitude_m,
        ]
        msg.yaw = self.target_yaw_rad
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        """Publish a command addressed to the configured PX4 vehicle."""
        msg = VehicleCommand()
        msg.timestamp = self._timestamp_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.target_system
        msg.target_component = self.target_component
        msg.source_system = self.source_system
        msg.source_component = self.source_component
        msg.from_external = True
        self.command_pub.publish(msg)

    def timer_callback(self) -> None:
        """Stream setpoints before requesting Offboard mode and arming."""
        if not self.position_received:
            return

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()
        self.counter += 1

        if self.counter == self.offboard_request_count:
            self.get_logger().info('Requesting OFFBOARD mode')
            # MAV_CMD_DO_SET_MODE: custom mode 6 is PX4 Offboard mode.
            self.publish_vehicle_command(176, 1.0, 6.0)

        if self.counter == self.arm_request_count:
            self.get_logger().info('Requesting vehicle arm')
            # MAV_CMD_COMPONENT_ARM_DISARM.
            self.publish_vehicle_command(400, 1.0, 0.0)


def main(args=None) -> None:
    """Run the Offboard takeoff node."""
    rclpy.init(args=args)
    node = OffboardTakeoff()

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
