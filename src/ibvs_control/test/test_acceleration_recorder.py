"""Synthetic node lifecycle: only the three requested artifacts are written."""

from gz.msgs10.pose_v_pb2 import Pose_V
from interception_interfaces.msg import ControlDebug
import rclpy

from ibvs_control.acceleration_recorder import AccelerationRecorder


def test_record_window_and_full_motion_plots(tmp_path):
    rclpy.init(args=['--ros-args', '-p', f'output_directory:={tmp_path}'])
    node = AccelerationRecorder()
    try:
        for sec in range(1, 6):
            control = ControlDebug()
            control.stamp.sec = sec
            control.control_state = 'ACTIVE' if 2 <= sec < 4 else 'LAND'
            control.valid = control.control_state == 'ACTIVE'
            control.acceleration_d.x = 2.
            control.impact_detected = sec >= 4
            node._control(control)
            message = Pose_V()
            message.header.stamp.sec = sec
            pose = message.pose.add()
            pose.name = node.vehicle
            pose.position.x = float(sec ** 2)
            node._pose(message)
            node._drain()
        assert node.record.origin == 2_000_000_000
        assert node.record.end == node.contact_time == 4_000_000_000
        assert len(node.velocity) == 4
        assert len(node.acceleration) == 3
        assert all(abs(sample[1][0] - 2.) < 1e-9 for sample in node.acceleration)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    runs = list(tmp_path.iterdir())
    assert runs == [node.directory]
    assert node.directory.is_dir()
    assert len(list(node.directory.iterdir())) == 3
    assert len(list(node.directory.glob('*.xlsx'))) == 1
    assert len(list(node.directory.glob('*.png'))) == 2
