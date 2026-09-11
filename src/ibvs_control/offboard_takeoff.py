import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition
)

class OffboardTakeoff(Node):

    def __init__(self):
        super().__init__('offboard_takeoff')

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_pub
        )

        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_pub
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_pub
        )

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.position_callback,
            qos_sub
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.position_received = False

        self.counter = 0
        self.timer = self.create_timer(0.1, self.timer_callback)

    def position_callback(self, msg):
        if not self.position_received:
            self.current_x = float(msg.x)
            self.current_y = float(msg.y)
            self.position_received = True

            self.get_logger().info(
                f'Initial position: x={self.current_x:.2f}, '
                f'y={self.current_y:.2f}'
            )

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = [
            self.current_x,
            self.current_y,
            -2.0
        ]

        msg.yaw = 0.0

        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.param1 = param1
        msg.param2 = param2

        msg.command = command

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1

        msg.from_external = True

        self.command_pub.publish(msg)

    def timer_callback(self):

        if not self.position_received:
            return

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        self.counter += 1

        # 先连续发送约 2 秒 setpoint
        if self.counter == 20:
            self.get_logger().info('Switching to OFFBOARD mode')

            # MAV_CMD_DO_SET_MODE
            self.publish_vehicle_command(
                176,
                1.0,
                6.0
            )

        # 稍后再解锁
        if self.counter == 25:
            self.get_logger().info('Arming vehicle')

            # MAV_CMD_COMPONENT_ARM_DISARM
            self.publish_vehicle_command(
                400,
                1.0,
                0.0
            )


def main(args=None):
    rclpy.init(args=args)

    node = OffboardTakeoff()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
