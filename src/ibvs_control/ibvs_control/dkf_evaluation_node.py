"""Read-only DKF evaluator; Gazebo truth never enters flight topics."""

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import threading
from uuid import uuid4

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from interception_interfaces.msg import ControlDebug, ObserverEvaluation
import numpy as np
from px4_msgs.msg import TimesyncStatus
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_control.dkf_evaluation import EvaluationCsv, STATE_NAMES, truth_between


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nsec)


class DkfEvaluationNode(Node):
    """Buffer timestamped world poses until each estimate can be bracketed."""

    def __init__(self):
        super().__init__('dkf_evaluation')
        defaults = {
            'world_name': 'default', 'vehicle_model': 'x500_mono_cam_0',
            'target_model': 'ibvs_target', 'px4_to_gazebo_offset_s': float('nan'),
            'output_directory': str(Path.home() / 'ibvs_ws' / 'test_results' / 'dkf'),
            'run_id': '',
            'camera_to_body_rotation': [0., 0., 1., -1., 0., 0., 0., -1., 0.],
            'camera_position_b_m': [0.14, 0., 0.],
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        config = {key: self.get_parameter(key).value for key in defaults}
        self.offset = config['px4_to_gazebo_offset_s']
        self.vehicle = config['vehicle_model']
        self.target = config['target_model']
        self.rotation = np.array(config['camera_to_body_rotation']).reshape(3, 3)
        self.position = np.array(config['camera_position_b_m'])
        run = config['run_id'] or (
            datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
        )
        self.csv = EvaluationCsv(Path(config['output_directory']) / run)
        # A null manual offset selects the existing DDS timesync status stream.
        config['px4_to_gazebo_offset_s'] = self.offset if np.isfinite(self.offset) else None
        config['state_order'] = STATE_NAMES
        config['velocity_truth'] = 'world-pose secant over recorded bracket (approximate)'
        config['bias_truth'] = 'unavailable'
        config['clock_mapping'] = (
            'manual constant' if np.isfinite(self.offset) else
            'DDS timestamp + TimesyncStatus.estimated_offset = Gazebo time'
        )
        self.metadata = config
        self.start_ns = None
        self.end_ns = None
        self.control_time_ns = None
        self.metadata.update({
            'recording_start_time_ns': None, 'recording_end_time_ns': None,
            'recording_window': 'first ACTIVE to first subsequent non-ACTIVE, inclusive',
        })
        self._save_metadata()
        self.time_offsets = deque(maxlen=256)
        self.poses = deque(maxlen=4096)
        self.pending = deque()
        self.lock = threading.Lock()
        self.gz = GazeboNode()
        topic = f"/world/{config['world_name']}/pose/info"
        if not self.gz.subscribe(Pose_V, topic, self._poses):
            raise RuntimeError(f'Cannot subscribe to {topic}')
        self.create_subscription(ObserverEvaluation, '/interception/observer/evaluation',
                                 self._estimate, 100)
        self.create_subscription(
            TimesyncStatus, '/fmu/out/timesync_status', self._timesync,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ControlDebug, '/interception/control/debug', self._control, 100,
        )
        self.create_timer(0.02, self._drain)
        self.create_timer(1.0, self.csv.summarize)
        self.get_logger().info(f'DKF evaluation: {self.csv.directory}')
        if not np.isfinite(self.offset):
            self.get_logger().info('Automatic clock mapping: /fmu/out/timesync_status')

    def _save_metadata(self):
        (self.csv.directory / 'metadata.json').write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + '\n')

    def _control(self, message):
        self.control_time_ns = int(message.stamp.sec) * 1_000_000_000 + message.stamp.nanosec
        if self.start_ns is None and message.control_state == 'ACTIVE':
            self.start_ns = self.control_time_ns
            self.metadata['recording_start_time_ns'] = self.start_ns
            self._save_metadata()
            self.get_logger().info('DKF 记录开始：进入 ACTIVE')
        elif self.start_ns is not None and self.end_ns is None and message.control_state != 'ACTIVE':
            self.end_ns = self.control_time_ns
            self.metadata.update({
                'recording_end_time_ns': self.end_ns,
                'recording_end_state': message.control_state,
                'recording_end_reason': message.reason,
            })
            self._save_metadata()
            self.get_logger().info(f'DKF 记录结束：{message.control_state}')
        self._drain()

    def _timesync(self, message):
        if message.source_protocol == TimesyncStatus.SOURCE_PROTOCOL_DDS:
            # PX4 serializes timestamp = hrt - estimated_offset; hrt follows GZ.
            self.time_offsets.append((int(message.timestamp), int(message.estimated_offset)))
            self._drain()

    def _offset_us(self, timestamp_us):
        if np.isfinite(self.offset):
            return int(round(self.offset * 1e6))
        # Use the preceding sender offset, not a future update or receipt time.
        for stamp, offset in reversed(self.time_offsets):
            if stamp <= timestamp_us:
                return offset
        return None

    def _poses(self, message):
        poses = {pose.name: pose for pose in message.pose}
        if self.vehicle not in poses or self.target not in poses:
            return
        u, t = poses[self.vehicle], poses[self.target]
        q = np.array([u.orientation.w, u.orientation.x,
                      u.orientation.y, u.orientation.z])
        q /= np.linalg.norm(q)
        sample = (stamp_ns(message.header.stamp),
                  np.array([u.position.x, u.position.y, u.position.z]), q,
                  np.array([t.position.x, t.position.y, t.position.z]))
        with self.lock:
            if self.poses and sample[0] <= self.poses[-1][0]:
                return
            self.poses.append(sample)

    def _estimate(self, message):
        self.pending.append(message)
        self._drain()

    def _drain(self, final=False):
        with self.lock:
            poses = list(self.poses)
        while self.pending:
            message = self.pending[0]
            state = message.state
            offset_us = self._offset_us(message.imu_timestamp_us)
            time = ((int(message.imu_timestamp_us) + offset_us) * 1000
                    if message.imu_timestamp_us and offset_us is not None else None)
            pub = int(state.stamp.sec) * 1_000_000_000 + state.stamp.nanosec
            window_time = time if time is not None else pub
            if self.end_ns is not None and window_time > self.end_ns:
                self.pending.popleft()
                continue
            # Wait for the ordered control timeline: a late end notification
            # must not allow recovery samples to enter the recorded interval.
            if not final and (self.control_time_ns is None or window_time > self.control_time_ns):
                if len(self.pending) < 4096:
                    return
                self.pending.popleft()
                continue
            upper = self.end_ns if self.end_ns is not None else self.control_time_ns
            if (self.start_ns is None or window_time < self.start_ns
                    or upper is None or window_time > upper):
                self.pending.popleft()
                continue
            usable = time is not None and state.initialized and state.valid
            if usable and not final and len(self.pending) < 4096 and (
                    not poses or time > poses[-1][0]):
                return
            self.pending.popleft()
            truth = np.full(18, np.nan)
            status = 'clock_unavailable' if offset_us is None else 'state_time_unavailable'
            before = after = ''
            if usable:
                status = 'truth_not_bracketed'
                times = [pose[0] for pose in poses]
                index = int(np.searchsorted(times, time, side='left'))
                if times and time == times[0]:
                    index = 1
                if 0 < index < len(poses):
                    before, after = poses[index - 1][0], poses[index][0]
                    truth = truth_between(poses[index - 1], poses[index], time,
                                          self.rotation, self.position)
                    status = 'pose_interpolated_velocity_secant_bias_unavailable'
            estimate = list(state.q)
            for vector in (state.p_r, state.v_r):
                estimate.extend([vector.x, vector.y, vector.z])
            estimate.extend(state.image_xy)
            for vector in (state.b_gyr, state.b_acc):
                estimate.extend([vector.x, vector.y, vector.z])
            if not state.initialized:
                estimate = np.full(18, np.nan)
            self.csv.write([pub, message.imu_timestamp_us, time if time is not None else '',
                            state.initialized, state.valid, state.reason, status,
                            before, after, offset_us if offset_us is not None else ''],
                           truth, estimate)

    def destroy_node(self):
        if self.start_ns is not None and self.end_ns is None:
            self.metadata['recording_end_status'] = 'interrupted; no transition out of ACTIVE received'
            self._save_metadata()
        self._drain(final=True)
        self.csv.close()
        self.get_logger().info(
            f'DKF 测试结果已保存，请打开：{self.csv.directory / "测试结果.html"}'
        )
        return super().destroy_node()


def main(args=None):
    """Run the evaluation session started automatically by the trial launch."""
    rclpy.init(args=args)
    node = DkfEvaluationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
