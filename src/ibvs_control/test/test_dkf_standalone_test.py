"""Verify independent motion, bootstrap and recorder boundaries without flight."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
from types import SimpleNamespace

from gz.msgs10.pose_v_pb2 import Pose_V
from interception_interfaces.msg import ObserverEvaluation, VisionFeature
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import numpy as np
from px4_msgs.msg import VehicleOdometry
import pytest
import rclpy
from rclpy.time import Time

from ibvs_control.dkf_evaluation_node import DkfEvaluationNode
from ibvs_control.dkf_evaluation_report import read_samples, snapshot_rows
from ibvs_control.dkf_standalone_test import DkfStandaloneTest, position_offset_enu
from ibvs_control.takeoff_state_machine import FlightPhase


class Clock:
    """Provide deterministic physical time for callbacks."""

    nanoseconds = 100_000_000_000

    def now(self):
        """Return the currently selected test time."""
        return Time(nanoseconds=self.nanoseconds)


@pytest.fixture
def runner(monkeypatch, request):
    test_case = getattr(request, 'param', 'B')
    rclpy.init(args=[
        '--ros-args', '-p', f'test_case:={test_case}', '-p', 'enable_flight_commands:=true',
        '-p', 'confirmation_token:=ENABLE_VISION_ONLY_CONTROL',
        '-p', 'target_altitude_m:=10.0', '-p', 'target_yaw_rad:=1.5707963267948966',
    ])
    node = DkfStandaloneTest()
    clock = Clock()
    monkeypatch.setattr(node, 'get_clock', lambda: clock)
    node.state_machine.phase = FlightPhase.HOLD
    node.state_machine.phase_entered_s = 100.0
    node.state_machine.home_x = node.state_machine.home_y = 0.0
    node.state_machine.home_z = 0.0
    node.state_machine.armed = True
    node.state_machine.landed = False
    # Leave safety checks to the existing takeoff tests; exercise actual actions.
    monkeypatch.setattr(node.state_machine, '_run_safety_checks', lambda now: None)
    node.state_machine.clock_started = True
    yield node, clock
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize(('runner', 'peak_times'), [
    ('B', (5.0, 3.75, 4.5)), ('C', (5.0, 2.5, 5.0)),
], indirect=['runner'])
def test_motion_frame_mapping_and_position_continuity(runner, monkeypatch, peak_times):
    node, clock = runner
    assert position_offset_enu(node.test_case, peak_times[0])[0] == pytest.approx(4.0)
    assert position_offset_enu(node.test_case, peak_times[1])[1] == pytest.approx(6.0)
    assert position_offset_enu(node.test_case, peak_times[2])[2] == pytest.approx(2.0)
    node.test_start_ns = clock.nanoseconds
    published = []
    monkeypatch.setattr(node.trajectory_pub, 'publish', published.append)
    for elapsed in (0.0, 3.75, 5.0, 60.0, 62.0):
        clock.nanoseconds = 100_000_000_000 + int(elapsed * 1e9)
        node.publish_trajectory_setpoint()
        east, north, up = position_offset_enu(node.test_case, min(elapsed, 60.0))
        assert list(published[-1].position) == pytest.approx([north, east, -10.0 - up])
        assert published[-1].yaw == pytest.approx(math.pi / 2)
    assert list(published[0].position) == pytest.approx([0.0, 0.0, -10.0])
    assert list(published[-1].position) == list(published[-2].position)
    assert position_offset_enu('A', 5.0) == (0., 0., 0.)


def test_c_figure_eight_crossing_lobes_and_altitude():
    half_diagonal = math.sqrt(2.0)
    for elapsed_s, expected in (
        (0.0, (0.0, 0.0, 0.0)),
        (2.5, (2.0 * half_diagonal, 6.0, half_diagonal)),
        (5.0, (4.0, 0.0, 2.0)),
        (7.5, (2.0 * half_diagonal, -6.0, half_diagonal)),
        (10.0, (0.0, 0.0, 0.0)),
        (12.5, (-2.0 * half_diagonal, 6.0, -half_diagonal)),
        (15.0, (-4.0, 0.0, -2.0)),
        (17.5, (-2.0 * half_diagonal, -6.0, -half_diagonal)),
        (20.0, (0.0, 0.0, 0.0)),
    ):
        assert position_offset_enu('C', elapsed_s) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('runner', ['D'], indirect=True)
def test_d_circle_geometry_speed_and_ned_mapping(runner, monkeypatch):
    node, clock = runner
    period_s = 6.0 * math.pi
    projected_radius = 6.0 / math.sqrt(2.0)
    node.test_start_ns = clock.nanoseconds
    published = []
    monkeypatch.setattr(node.trajectory_pub, 'publish', published.append)
    for fraction, expected in (
        (0.0, (0.0, 0.0, 0.0)),
        (0.25, (projected_radius, 6.0, -projected_radius)),
        (0.5, (2.0 * projected_radius, 0.0, -2.0 * projected_radius)),
        (0.75, (projected_radius, -6.0, -projected_radius)),
        (1.0, (0.0, 0.0, 0.0)),
    ):
        elapsed_s = fraction * period_s
        assert position_offset_enu('D', elapsed_s) == pytest.approx(expected, abs=1e-9)
        clock.nanoseconds = 100_000_000_000 + int(elapsed_s * 1e9)
        node.publish_trajectory_setpoint()
        east, north, up = expected
        assert list(published[-1].position) == pytest.approx(
            [north, east, -10.0 - up], abs=1e-6,
        )
        assert published[-1].yaw == pytest.approx(math.pi / 2)
    for elapsed_s in (1.0, 4.0, 9.0, 17.0):
        before = np.array(position_offset_enu('D', elapsed_s - 1e-4))
        after = np.array(position_offset_enu('D', elapsed_s + 1e-4))
        assert np.linalg.norm((after - before) / 2e-4) == pytest.approx(2.0)
    for elapsed_s in (60.0, 62.0):
        clock.nanoseconds = 100_000_000_000 + int(elapsed_s * 1e9)
        node.publish_trajectory_setpoint()
    assert list(published[-1].position) == list(published[-2].position)


@pytest.mark.parametrize('runner', ['A', 'B', 'C', 'D'], indirect=True)
def test_bootstrap_starts_once_and_timer_never_runs_interception(runner, monkeypatch):
    node, clock = runner
    assert node.rates_pub is None
    monkeypatch.setattr(node, '_update_control', lambda _: pytest.fail('interception ran'))
    monkeypatch.setattr(node, '_run_enabled_step', lambda *_: pytest.fail('interception ran'))
    statuses, resets, bootstraps = [], [], []
    monkeypatch.setattr(node.debug_pub, 'publish', statuses.append)
    monkeypatch.setattr(node.observer_reset_pub, 'publish', resets.append)
    monkeypatch.setattr(node.observer_bootstrap_pub, 'publish', bootstraps.append)
    node.timer_callback()
    assert node.test_start_ns is None
    assert statuses[-1].control_state == 'HOLD'
    odometry = VehicleOdometry()
    odometry.position = [0.0, 0.0, -10.0]
    odometry.q = [1.0, 0.0, 0.0, 0.0]
    node._odometry_callback(odometry)
    feature = VisionFeature()
    feature.capture_stamp = clock.now().to_msg()
    feature.width_norm = 0.26 / 30.0
    feature.valid = True
    node._feature_callback(feature)
    node.timer_callback()
    assert node.test_start_ns == clock.nanoseconds
    assert statuses[-1].control_state == 'ACTIVE'
    assert len(bootstraps) == len(resets) == 1
    covariance = np.array(bootstraps[0].relative_covariance).reshape((6, 6))
    assert np.linalg.eigvalsh(covariance)[0] > 0
    clock.nanoseconds += int(node.duration_s * 1e9) - 20_000_000
    node.timer_callback()  # No observer output: prescribed motion continues.
    assert statuses[-1].control_state == 'ACTIVE'
    clock.nanoseconds += 20_000_000
    node.timer_callback()
    assert statuses[-1].control_state == 'LAND'
    assert node.state_machine.phase == FlightPhase.LAND
    assert len(bootstraps) == len(resets) == 1


@pytest.mark.parametrize('test_case', ['A', 'B', 'C', 'D'])
def test_launch_reuses_observer_and_private_recording_window(test_case, tmp_path):
    path = Path(__file__).parents[1] / 'launch' / 'dkf_standalone_test.launch.py'
    spec = spec_from_file_location('dkf_test_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations.update({
        'test_case': test_case, 'trial_id': f'case_{test_case}',
        'output_directory': str(tmp_path),
    })
    actions = module._launch_test(context)
    nodes = {action.node_executable: action for action in actions if isinstance(action, Node)}
    assert set(nodes) == {'dkf_standalone_test', 'dkf_evaluation', 'paper_state_observer'}
    flight, = evaluate_parameters(context, nodes['dkf_standalone_test']._Node__parameters)
    assert flight['test_case'] == test_case
    assert flight['target_altitude_m'] == 10.0
    assert flight['target_yaw_rad'] == math.pi / 2
    assert flight['debug_topic'] == '/interception/dkf_test/status'
    assert flight['target_diameter_m'] == 0.26
    evaluation, = evaluate_parameters(context, nodes['dkf_evaluation']._Node__parameters)
    assert evaluation['test_case'] == test_case
    assert evaluation['run_id'] == f'case_{test_case}'
    assert evaluation['output_directory'] == str(tmp_path)


@pytest.mark.parametrize('test_case', ['C', 'D'])
def test_private_window_still_produces_existing_half_second_table(tmp_path, test_case):
    rclpy.init(args=[
        '--ros-args', '-p', f'output_directory:={tmp_path}', '-p', f'test_case:={test_case}',
        '-p', 'run_id:=standalone', '-p', 'px4_to_gazebo_offset_s:=0.0',
    ])
    evaluator = DkfEvaluationNode()
    directory = evaluator.csv.directory
    try:
        for second in (99, 102):
            poses = Pose_V()
            poses.header.stamp.sec = second
            for name, x in (('x500_mono_cam_0', 0), ('ibvs_target', 40)):
                pose = poses.pose.add()
                pose.name = name
                pose.position.x = x
                pose.orientation.w = 1.0
            evaluator._poses(poses)
        for elapsed, phase in ((0., 'ACTIVE'), (0.5, 'ACTIVE'), (1., 'LAND')):
            status = SimpleNamespace(
                stamp=Time(nanoseconds=int((100 + elapsed) * 1e9)).to_msg(),
                control_state=phase, reason=f'DKF test {test_case}',
            )
            evaluator._control(status)
            output = ObserverEvaluation()
            output.imu_timestamp_us = int((100 + elapsed) * 1e6)
            output.state.initialized = output.state.valid = True
            output.state.q = [1., 0., 0., 0.]
            evaluator._estimate(output)
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()
    rows = list(csv.DictReader((directory / 'states.csv').open()))
    assert len(rows) == 3
    _, samples, _ = read_samples(directory)
    assert [row[0] for row in snapshot_rows(samples)] == ['0.000', '0.500', '1.000']
    assert (directory / '每0.5秒对比.xlsx').exists()
    report = (directory / '测试结果.html').read_text()
    assert f'DKF 独立初测工况：{test_case}' in report
    assert '时间从独立测试开始计为 0 秒' in report
