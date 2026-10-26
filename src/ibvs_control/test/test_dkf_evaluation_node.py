"""Synthetic ROS/Gazebo-message checks; no running simulator is required."""

import csv

from gz.msgs10.pose_v_pb2 import Pose_V
from interception_interfaces.msg import ControlDebug, ObserverEvaluation
from px4_msgs.msg import TimesyncStatus
import rclpy

from ibvs_control.dkf_evaluation_node import DkfEvaluationNode
from ibvs_control.paper_state_observer_node import PaperStateObserverNode


def test_evaluator_brackets_state_time_and_keeps_unavailable_outputs(tmp_path):
    rclpy.init(args=[
        '--ros-args', '-p', f'output_directory:={tmp_path}',
        '-p', 'px4_to_gazebo_offset_s:=0.0',
    ])
    node = DkfEvaluationNode()
    directory = node.csv.directory
    try:
        control = ControlDebug()
        control.stamp.sec = 1
        control.control_state = 'ACTIVE'
        node._control(control)
        control.stamp.sec = 3
        node._control(control)
        output = ObserverEvaluation()
        output.imu_timestamp_us = 2_000_000
        output.state.stamp.sec = 9  # Publication time must not select truth.
        output.state.initialized = True
        output.state.valid = True
        output.state.q = [1., 0., 0., 0.]
        node._estimate(output)
        assert len(node.pending) == 1
        for sec in (1, 3):
            poses = Pose_V()
            poses.header.stamp.sec = sec
            vehicle = poses.pose.add()
            vehicle.name = 'x500_mono_cam_0'
            vehicle.orientation.w = 1
            vehicle.position.x = sec
            target = poses.pose.add()
            target.name = 'ibvs_target'
            target.position.x = 12
            target.orientation.w = 1
            node._poses(poses)
        node._drain()
        unavailable = ObserverEvaluation()
        unavailable.state.stamp.sec = 2
        node._estimate(unavailable)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    rows = list(csv.DictReader((directory / 'states.csv').open()))
    assert len(rows) == 2
    assert rows[0]['state_time_ns'] == '2000000000'
    assert rows[0]['publish_time_ns'] == '9000000000'
    assert float(rows[0]['p_r_e_true']) == -10
    assert float(rows[0]['v_r_e_true']) == 1
    assert rows[1]['truth_status'] == 'state_time_unavailable'


def test_evaluation_copy_preserves_original_message_and_imu_time(monkeypatch):
    rclpy.init()
    node = PaperStateObserverNode()
    try:
        original, copies = [], []
        monkeypatch.setattr(node.publisher, 'publish', original.append)
        monkeypatch.setattr(node.evaluation_publisher, 'get_subscription_count', lambda: 1)
        monkeypatch.setattr(node.evaluation_publisher, 'publish', copies.append)
        node.last_imu_timestamp_us = 123456
        node._publish_uninitialized('test')
        assert copies[0].state == original[0]
        assert copies[0].imu_timestamp_us == 123456
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_automatic_dds_clock_mapping_and_final_report(tmp_path):
    rclpy.init(args=['--ros-args', '-p', f'output_directory:={tmp_path}',
                    '-p', 'run_id:=automatic_trial'])
    node = DkfEvaluationNode()
    directory = node.csv.directory
    try:
        # Simulate DDS's epoch translation, preserving integer microseconds.
        epoch_us = 1_790_000_000_000_000
        sync = TimesyncStatus()
        sync.source_protocol = TimesyncStatus.SOURCE_PROTOCOL_MAVLINK
        sync.timestamp = epoch_us + 1_000_000
        sync.estimated_offset = -epoch_us
        node._timesync(sync)
        assert node._offset_us(epoch_us + 2_000_000) is None
        sync.source_protocol = TimesyncStatus.SOURCE_PROTOCOL_DDS
        node._timesync(sync)
        assert node._offset_us(epoch_us) is None
        for sec in (1, 3):
            poses = Pose_V()
            poses.header.stamp.sec = sec
            for name, x in (('x500_mono_cam_0', sec), ('ibvs_target', 12)):
                pose = poses.pose.add()
                pose.name = name
                pose.position.x = x
                pose.orientation.w = 1
            node._poses(poses)
        control = ControlDebug()
        control.stamp.sec = 1
        control.control_state = 'ACTIVE'
        node._control(control)
        control.stamp.sec = 3
        node._control(control)
        output = ObserverEvaluation()
        output.imu_timestamp_us = epoch_us + 2_000_000
        output.state.initialized = output.state.valid = True
        output.state.q = [1., 0., 0., 0.]
        output.state.p_r.x = -9.
        node._estimate(output)
        # A future offset cannot change this sample's selected physical time.
        sync.timestamp = epoch_us + 4_000_000
        sync.estimated_offset = -epoch_us + 123
        node._timesync(sync)
        assert node._offset_us(output.imu_timestamp_us) == -epoch_us
    finally:
        node.destroy_node()
        rclpy.shutdown()
    assert directory == tmp_path / 'automatic_trial'
    row, = csv.DictReader((directory / 'states.csv').open())
    assert row['state_time_ns'] == '2000000000'
    assert row['clock_offset_us'] == str(-epoch_us)
    assert float(row['p_r_e_error']) == 1.
    metrics = {row['state']: row for row in csv.DictReader(
        (directory / 'metrics.csv').open())}
    assert float(metrics['p_r_e']['rmse']) == 1.
    assert metrics['b_acc_f']['count'] == '0'


def test_only_active_interval_is_recorded_including_both_endpoints(tmp_path):
    import json

    rclpy.init(args=['--ros-args', '-p', f'output_directory:={tmp_path}',
                    '-p', 'px4_to_gazebo_offset_s:=0.0'])
    node = DkfEvaluationNode()
    directory = node.csv.directory

    def control(time_us, phase):
        message = ControlDebug()
        message.stamp.sec, message.stamp.nanosec = divmod(time_us * 1000, 1_000_000_000)
        message.control_state = phase
        node._control(message)

    def estimate(time_us):
        message = ObserverEvaluation()
        message.imu_timestamp_us = time_us
        message.state.initialized = message.state.valid = True
        message.state.q = [1., 0., 0., 0.]
        node._estimate(message)

    try:
        for sec in (0, 4):
            message = Pose_V()
            message.header.stamp.sec = sec
            for name, x in (('x500_mono_cam_0', sec), ('ibvs_target', 12)):
                pose = message.pose.add()
                pose.name = name
                pose.position.x = x
                pose.orientation.w = 1
            node._poses(message)
        estimate(900_000)
        control(900_000, 'PRESTREAM')
        estimate(1_000_000)  # Arrives before the ACTIVE notification.
        control(1_000_000, 'ACTIVE')
        estimate(1_500_000)
        control(1_500_000, 'ACTIVE')
        estimate(2_000_000)
        estimate(2_230_000)  # Exact end sample arrives before the end notification.
        estimate(2_300_000)  # Must wait; cannot leak into CSV before a late end.
        control(2_230_000, 'ABORT')
        estimate(2_400_000)
        control(3_000_000, 'LAND')
        assert not node.pending
    finally:
        node.destroy_node()
        rclpy.shutdown()
    rows = list(csv.DictReader((directory / 'states.csv').open()))
    assert [int(row['state_time_ns']) for row in rows] == [
        1_000_000_000, 1_500_000_000, 2_000_000_000, 2_230_000_000]
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['recording_start_time_ns'] == 1_000_000_000
    assert metadata['recording_end_time_ns'] == 2_230_000_000
    from ibvs_control.dkf_evaluation_report import read_samples, snapshot_rows
    _, samples, origin = read_samples(directory)
    snapshots = list(snapshot_rows(samples))
    assert origin == 1_000_000_000
    assert [row[0] for row in snapshots] == ['0.0', '0.5', '1.0', '1.230']
    assert snapshots[-1][1] == '1.230'
    metrics = list(csv.DictReader((directory / 'metrics.csv').open()))
    assert metrics[0]['count'] == '4'
