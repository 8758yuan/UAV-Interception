"""Publish controller inputs from one native Gazebo physics-step snapshot."""

import threading

from gz.msgs10.any_pb2 import Any
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.param_v_pb2 import Param_V
from gz.transport13 import Node as GazeboNode
from interception_interfaces.msg import ControllerTruth
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from ibvs_control.frames import quaternion_wxyz_to_rotation


class ControllerTruthState(Node):
    """Use physics pose and velocity directly, without differencing or filtering."""

    def __init__(self):
        super().__init__('controller_truth_state')
        defaults = {
            'world_name': 'default',
            'state_topic': '/interception/controller_test/state',
            'stop_motion_on_shutdown': False,
            'camera_to_body_rotation': [0., 0., 1., -1., 0., 0., 0., -1., 0.],
            'camera_position_b_m': [0.14, 0., 0.],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        config = {name: self.get_parameter(name).value for name in defaults}
        self.stop_motion = bool(config['stop_motion_on_shutdown'])
        self.stop_service = f"/world/{config['world_name']}/paper_target/stop"
        self.rotation = np.asarray(config['camera_to_body_rotation']).reshape(3, 3)
        self.position = np.asarray(config['camera_position_b_m'])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(ControllerTruth, config['state_topic'], qos)
        self.pending = None
        self.closing = False
        self.lock = threading.Lock()
        self.publish_guard = self.create_guard_condition(self._publish)
        self.gz = GazeboNode()
        self.physics_topic = f"/world/{config['world_name']}/controller_test/state"
        if not self.gz.subscribe(Param_V, self.physics_topic, self._snapshot):
            raise RuntimeError(f'Cannot subscribe to {self.physics_topic}')
        self.get_logger().info(
            f'Native Gazebo physics snapshots -> {config["state_topic"]}; '
            'position, velocity, attitude and image share one simulation timestamp'
        )

    def _snapshot(self, message):
        output = ControllerTruth()
        state = output.state
        state.stamp.sec = message.header.stamp.sec
        state.stamp.nanosec = message.header.stamp.nsec
        # These observer-only quantities are not measured or used by control.
        state.b_gyr.x = state.b_gyr.y = state.b_gyr.z = float('nan')
        state.b_acc.x = state.b_acc.y = state.b_acc.z = float('nan')
        state.covariance = [float('nan')] * 324
        expected = {'vehicle_pose': Any.POSE3D, 'target_pose': Any.POSE3D,
                    'vehicle_velocity': Any.VECTOR3D, 'target_velocity': Any.VECTOR3D,
                    'vehicle_angular_velocity': Any.VECTOR3D}
        if len(message.param) != 1 or any(
            key not in message.param[0].params or message.param[0].params[key].type != kind
            for key, kind in expected.items()
        ):
            state.reason = 'gazebo_physics_snapshot_incomplete'
            self._queue_snapshot(output)
            return
        fields = message.param[0].params
        pose = fields['vehicle_pose'].pose3d_value
        target = fields['target_pose'].pose3d_value.position
        q = np.array([pose.orientation.w, pose.orientation.x,
                      pose.orientation.y, pose.orientation.z])

        def vector(value):
            return np.array([value.x, value.y, value.z])

        pu, pt = vector(pose.position), vector(target)
        vu = vector(fields['vehicle_velocity'].vector3d_value)
        vt = vector(fields['target_velocity'].vector3d_value)
        angular = vector(fields['vehicle_angular_velocity'].vector3d_value)
        state.initialized = True
        norm = float(np.linalg.norm(q))
        if (not np.isfinite(np.concatenate([q, pu, pt, vu, vt, angular])).all()
                or not np.isfinite(norm) or norm <= 1e-12):
            state.reason = 'gazebo_physics_snapshot_nonfinite_or_invalid_quaternion'
            self._queue_snapshot(output)
            return
        q /= norm
        rotation = quaternion_wxyz_to_rotation(q)
        state.q = q.tolist()
        state.p_r.x, state.p_r.y, state.p_r.z = (pu - pt).tolist()
        state.v_r.x, state.v_r.y, state.v_r.z = (vu - vt).tolist()
        output.vehicle_position_e.x, output.vehicle_position_e.y, output.vehicle_position_e.z = pu
        output.vehicle_velocity_e.x, output.vehicle_velocity_e.y, output.vehicle_velocity_e.z = vu
        omega_b = rotation.T @ angular
        (output.vehicle_angular_velocity_b.x, output.vehicle_angular_velocity_b.y,
         output.vehicle_angular_velocity_b.z) = omega_b
        point = self.rotation.T @ (rotation.T @ (pt - pu) - self.position)
        state.valid = bool(point[2] > 0.0)
        state.image_xy = (point[:2] / point[2]).tolist() if state.valid else [float('nan')] * 2
        state.reason = 'gazebo_native_physics_snapshot' if state.valid else 'target_behind_camera'
        self._queue_snapshot(output)

    def _queue_snapshot(self, output):
        # ROS publication belongs to its executor, not the Gazebo callback thread.
        with self.lock:
            if not self.closing:
                self.pending = output
                self.publish_guard.trigger()

    def _publish(self):
        with self.lock:
            output, self.pending = self.pending, None
        # Publish a physical sample once; stopped input cannot create a fresh heartbeat.
        if output is not None:
            self.publisher.publish(output)

    def destroy_node(self):
        with self.lock:
            self.closing = True
            self.pending = None
        self.gz.unsubscribe(self.physics_topic)
        if self.stop_motion:
            self.gz.request(self.stop_service, Empty(), Empty, Boolean, 100)
        return super().destroy_node()


def main(args=None):
    """Bridge native snapshots until ROS shutdown; never send PX4 commands."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ControllerTruthState()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
