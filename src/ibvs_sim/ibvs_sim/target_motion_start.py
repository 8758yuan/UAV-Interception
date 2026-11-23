"""Start the simulated target on the first ACTIVE interception state."""

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.transport13 import Node as GazeboNode

from interception_interfaces.msg import ControlDebug
import rclpy
from rclpy.node import Node


class TargetMotionStart(Node):
    """Relay the existing interception state to the target's start service."""

    def __init__(self) -> None:
        super().__init__('target_motion_start')
        self.declare_parameter('world', 'default')
        world = self.get_parameter('world').value
        self.service = f'/world/{world}/paper_target/start'
        self.gz_node = GazeboNode()
        self.started = False
        self.create_subscription(
            ControlDebug, '/interception/control/debug', self._control_callback, 10,
        )
        self.get_logger().info('Target held at its initial position; waiting for interception ACTIVE')

    def _control_callback(self, message: ControlDebug) -> None:
        if self.started or message.control_state != 'ACTIVE':
            return
        executed, response = self.gz_node.request(
            self.service, Empty(), Empty, Boolean, 1000,
        )
        if not executed or not response.data:
            self.get_logger().warning('Gazebo did not acknowledge target motion start')
            return
        self.started = True
        self.get_logger().info('Interception ACTIVE: target motion started')


def main(args=None) -> None:
    """Run the simulation-only interception start relay."""
    rclpy.init(args=args)
    node = TargetMotionStart()
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
