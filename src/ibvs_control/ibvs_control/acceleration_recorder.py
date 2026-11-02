"""Record motion without starting the DKF report pipeline or sending commands."""

from collections import deque
from datetime import datetime
from pathlib import Path
import threading
from uuid import uuid4

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from interception_interfaces.msg import ControlDebug
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ibvs_control.acceleration_record import AccelerationRecord, pose_acceleration


class AccelerationRecorder(Node):
    """Save one workbook and two full-rate motion plots per simulation."""

    def __init__(self):
        super().__init__('acceleration_recorder')
        defaults = {
            'world_name': 'default', 'vehicle_model': 'x500_mono_cam_0',
            'output_directory': str(Path.home() / 'ibvs_ws/results/acceleration'),
            'trial_id': '', 'debug_topic': '/interception/control/debug',
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        config = {key: self.get_parameter(key).value for key in defaults}
        run = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
        self.prefix = (Path(config['trial_id']).name + '_' if config['trial_id'] else '') + run
        self.directory = Path(config['output_directory']) / self.prefix
        self.directory.mkdir(parents=True, exist_ok=False)
        self.record = AccelerationRecord(self.directory, self.prefix + '_')
        self.vehicle = config['vehicle_model']
        self.lock = threading.Lock()
        self.pending = deque()
        self.recent = deque(maxlen=3)
        self.velocity = []
        self.acceleration = []
        self.control_time = 0
        self.contact_time = None
        self.gz = GazeboNode()
        self.topic = f"/world/{config['world_name']}/pose/info"
        if not self.gz.subscribe(Pose_V, self.topic, self._pose):
            raise RuntimeError(f'Cannot subscribe to {self.topic}')
        self.create_subscription(ControlDebug, config['debug_topic'], self._control, 100)
        self.create_timer(.02, self._drain)
        self.create_timer(.5, self._save)
        self.get_logger().info(f'Motion results: {self.directory}')

    def _pose(self, message):
        for pose in message.pose:
            if pose.name == self.vehicle:
                time = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
                with self.lock:
                    self.pending.append((time, np.array([
                        pose.position.x, pose.position.y, pose.position.z])))
                break

    def _drain(self):
        with self.lock:
            samples = list(self.pending)
            self.pending.clear()
        for sample in samples:
            if self.recent and sample[0] <= self.recent[-1][0]:
                continue
            self.recent.append(sample)
            if len(self.recent) >= 2:
                (t0, p0), (t1, p1) = list(self.recent)[-2:]
                self.velocity.append(((t0 + t1) // 2, (p1 - p0) / ((t1 - t0) * 1e-9)))
            if len(self.recent) == 3:
                time, value = pose_acceleration(self.recent)
                self.acceleration.append((time, value))
                self.record.add(1, time, value)

    def _control(self, message):
        self.control_time = message.stamp.sec * 1_000_000_000 + message.stamp.nanosec
        if self.record.origin is None and message.control_state == 'ACTIVE':
            self.record.origin = self.control_time
        elif (self.record.origin is not None and self.record.end is None
              and message.control_state != 'ACTIVE'):
            self.record.end = self.control_time
        if message.impact_detected and self.contact_time is None:
            self.contact_time = self.control_time
        if message.valid:
            value = message.acceleration_d
            self.record.add(0, self.control_time, [value.x, value.y, value.z])
        self._drain()

    def _save(self):
        self.record.save(self.control_time)

    def destroy_node(self):
        self.gz.unsubscribe(self.topic)
        self._drain()
        self._save()
        self.record.plot_motion(self.velocity, self.acceleration, self.contact_time)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AccelerationRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
