"""Test-only telemetry adapter around the unchanged production controller."""

from types import SimpleNamespace

from interception_interfaces.msg import ControllerTruth, ObserverState
import numpy as np
from px4_msgs.msg import VehicleOdometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions

from ibvs_control.frames import (
    enu_to_ned, flu_to_frd, quaternion_wxyz_to_rotation, rotation_to_quaternion_wxyz,
)
from ibvs_control.vision_interception_coordinator import VisionInterceptionCoordinator


class ControllerTruthCoordinator(VisionInterceptionCoordinator):
    """Feed coherent native states through the existing controller input fields."""

    def __init__(self):
        self.physics_sample = None
        super().__init__()
        self.get_logger().info('Controller test uses native attitude and own velocity as well')

    def create_subscription(self, msg_type, topic, callback, qos_profile, **kwargs):
        """Adapt the observer input type only in this dedicated test executable."""
        if msg_type is ObserverState:
            msg_type, callback = ControllerTruth, self._truth_callback
        return super().create_subscription(msg_type, topic, callback, qos_profile, **kwargs)

    def _truth_callback(self, sample):
        self.physics_sample = sample if sample.state.initialized and sample.state.valid else None
        if self.physics_sample is not None:
            state = sample.state
            # Retain native doubles instead of quantizing through PX4 float32 fields.
            odometry = SimpleNamespace()
            odometry.timestamp = odometry.timestamp_sample = (
                int(state.stamp.sec) * 1_000_000 + int(state.stamp.nanosec) // 1000
            )
            rotation = quaternion_wxyz_to_rotation(state.q)
            enu_to_ned_rotation = np.array(((0., 1., 0.), (1., 0., 0.), (0., 0., -1.)))
            odometry.q = rotation_to_quaternion_wxyz(
                enu_to_ned_rotation @ rotation @ np.diag((1., -1., -1.))
            )
            odometry.pose_frame = VehicleOdometry.POSE_FRAME_NED
            odometry.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            odometry.position = enu_to_ned((sample.vehicle_position_e.x,
                                           sample.vehicle_position_e.y,
                                           sample.vehicle_position_e.z))
            odometry.velocity = self._native_velocity_ned()
            odometry.angular_velocity = flu_to_frd((sample.vehicle_angular_velocity_b.x,
                                                   sample.vehicle_angular_velocity_b.y,
                                                   sample.vehicle_angular_velocity_b.z))
            super()._odometry_callback(odometry)
            self.vehicle_velocity_ned = np.asarray(odometry.velocity)
        super()._observer_callback(sample.state)

    def _native_velocity_ned(self):
        velocity = self.physics_sample.vehicle_velocity_e
        return enu_to_ned((velocity.x, velocity.y, velocity.z))

    def _odometry_callback(self, _message):
        """PX4 estimator odometry cannot replace the test's native snapshot."""

    def position_callback(self, message):
        """Keep takeoff/safety telemetry while preserving native control velocity."""
        super().position_callback(message)
        if self.physics_sample is not None:
            self.vehicle_velocity_ned = np.asarray(self._native_velocity_ned())


def main(args=None):
    """Run the existing flight flow with test-only state inputs."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ControllerTruthCoordinator()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
