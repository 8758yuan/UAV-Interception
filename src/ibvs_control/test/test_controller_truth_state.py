"""Synthetic geometry and ROS data-flow checks for the isolated controller test."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from gz.msgs10.any_pb2 import Any
from gz.msgs10.param_v_pb2 import Param_V
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import numpy as np
from interception_interfaces.msg import ControllerTruth
from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
import yaml

from ibvs_control.controller_truth_state import ControllerTruthState
from ibvs_control.controller_truth_coordinator import ControllerTruthCoordinator
from ibvs_control.frames import (
    enu_to_ned, px4_quaternion_to_enu_flu_rotation, rotation_to_quaternion_wxyz,
)
from ibvs_control import vision_interception_coordinator
from ibvs_control.so3_controller import compute_drag_force_e, compute_outer_loop
from ibvs_control.so3_controller import rodrigues
from ibvs_control.vision_interception_coordinator import VisionInterceptionCoordinator


def physics(time_ns, vehicle=(0., 0., 0.), target=(12., 0., 0.),
            q=(1., 0., 0., 0.), vu=(0., 0., 0.), vt=(0., 0., 0.), omega=(0., 0., 0.)):
    message = Param_V()
    message.header.stamp.sec, message.header.stamp.nsec = divmod(time_ns, 1_000_000_000)
    fields = message.param.add().params
    for name, position, quaternion in (('vehicle_pose', vehicle, q),
                                       ('target_pose', target, (1., 0., 0., 0.))):
        fields[name].type = Any.POSE3D
        pose = fields[name].pose3d_value
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion
    for name, values in (('vehicle_velocity', vu), ('target_velocity', vt),
                         ('vehicle_angular_velocity', omega)):
        fields[name].type = Any.VECTOR3D
        vector = fields[name].vector3d_value
        vector.x, vector.y, vector.z = values
    return message


@pytest.fixture
def truth_node():
    rclpy.init(args=[
        '--ros-args', '-p', 'observer_topic:=/interception/controller_test/state',
    ])
    node = ControllerTruthState()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_single_physics_sample_supplies_native_velocity_without_differencing(
    truth_node, monkeypatch,
):
    node = truth_node
    published = []
    monkeypatch.setattr(node.publisher, 'publish', published.append)
    rotation = rodrigues((0., 0., 1.), 0.3) @ rodrigues((1., 0., 0.), 0.2)
    q = np.asarray(rotation_to_quaternion_wxyz(rotation))
    pu, pt = np.array([1., -2., 3.]), np.array([15., 2., 7.])
    vu, vt = np.array([0.3, -0.7, 0.5]), np.array([-2., 0.4, 1.1])
    omega = np.array([0.2, -0.1, 0.4])
    node._snapshot(physics(1_200_000_000, pu, pt, -q, vu, vt, omega))
    node._publish()
    output, = published
    state = output.state
    assert state.initialized and state.valid
    assert state.stamp.sec == 1 and state.stamp.nanosec == 200_000_000
    np.testing.assert_allclose([state.p_r.x, state.p_r.y, state.p_r.z], pu - pt)
    np.testing.assert_allclose([state.v_r.x, state.v_r.y, state.v_r.z], vu - vt)
    np.testing.assert_allclose([output.vehicle_velocity_e.x, output.vehicle_velocity_e.y,
                                output.vehicle_velocity_e.z], vu, atol=1e-14)
    np.testing.assert_allclose([output.vehicle_angular_velocity_b.x,
                                output.vehicle_angular_velocity_b.y,
                                output.vehicle_angular_velocity_b.z], rotation.T @ omega)
    pc = node.rotation.T @ (rotation.T @ (pt - pu) - node.position)
    np.testing.assert_allclose(state.image_xy, pc[:2] / pc[2])
    assert np.isnan(state.b_gyr.x) and np.isnan(state.covariance).all()
    assert not list(node.timers)
    # A position jump and irregular interval do not alter the supplied native velocity.
    node._snapshot(physics(1_207_000_000, pu, pt + [3., 0., 0.], q, vu, vt, omega))
    node._publish()
    assert published[-1].state.v_r == state.v_r
    node._publish()
    assert len(published) == 2


@pytest.mark.parametrize('target', [(-12., 0., 0.), (0.14, 0., 0.)])
def test_nonpositive_camera_depth_is_invalid(truth_node, monkeypatch, target):
    published = []
    monkeypatch.setattr(truth_node.publisher, 'publish', published.append)
    truth_node._snapshot(physics(1_000_000_000, target=target))
    truth_node._publish()
    assert published[-1].state.initialized
    assert not published[-1].state.valid
    assert np.isnan(published[-1].state.image_xy).all()


def test_bad_native_data_and_clock_restart_do_not_use_old_state(truth_node, monkeypatch):
    published = []
    monkeypatch.setattr(truth_node.publisher, 'publish', published.append)
    truth_node._snapshot(physics(1_000_000_000, q=(0., 0., 0., 0.)))
    truth_node._publish()
    assert not published[-1].state.valid
    truth_node._snapshot(physics(1_020_000_000, vu=(float('nan'), 0., 0.)))
    truth_node._publish()
    assert not published[-1].state.valid
    truth_node._snapshot(physics(500_000_000, vu=(1., 2., 3.)))
    truth_node._publish()
    assert published[-1].state.valid
    assert published[-1].state.stamp.nanosec == 500_000_000
    message = physics(520_000_000)
    del message.param[0].params['target_velocity']
    truth_node._snapshot(message)
    truth_node._publish()
    assert not published[-1].state.valid
    assert published[-1].state.reason == 'gazebo_physics_snapshot_incomplete'


def test_native_snapshot_drives_existing_law_without_px4_state_estimates(
    truth_node, monkeypatch,
):
    coordinator = ControllerTruthCoordinator()
    assert (
        ControllerTruthCoordinator._update_control
        is VisionInterceptionCoordinator._update_control
    )
    executor = SingleThreadedExecutor()
    executor.add_node(coordinator)
    executor.add_node(truth_node)
    try:
        assert not coordinator.command_output_enabled
        deadline = time.monotonic() + 2.0
        while not truth_node.publisher.get_subscription_count() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        vu = (1.123456789012, -0.345678901234, 0.234567890123)
        rotation = rodrigues((0., 0., 1.), 0.1) @ rodrigues((0., 1., 0.), -0.07)
        q = rotation_to_quaternion_wxyz(rotation)
        truth_node._snapshot(physics(1_020_000_000, target=(12., 1., 0.5), q=q, vu=vu))
        deadline = time.monotonic() + 2.0
        while coordinator.observer_state is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert coordinator.observer_state is not None
        assert coordinator.observer_state.valid
        wrong_odometry = VehicleOdometry()
        wrong_odometry.q = [0., 1., 0., 0.]
        wrong_odometry.velocity = [99., 99., 99.]
        coordinator._odometry_callback(wrong_odometry)
        wrong_position = VehicleLocalPosition()
        wrong_position.vx = wrong_position.vy = wrong_position.vz = 99.
        coordinator.position_callback(wrong_position)
        np.testing.assert_allclose(coordinator.vehicle_velocity_ned, enu_to_ned(vu), atol=1e-14)
        np.testing.assert_allclose(px4_quaternion_to_enu_flu_rotation(coordinator.odometry.q),
                                   rotation, atol=1e-14)
        captured = []

        def capture(*args, **kwargs):
            captured.append((args, kwargs))
            return compute_outer_loop(*args, **kwargs)

        monkeypatch.setattr(vision_interception_coordinator, 'compute_outer_loop', capture)
        coordinator.last_outer_ns = None
        coordinator._update_control(coordinator.get_clock().now().nanoseconds)
        assert coordinator.px4_command is not None, coordinator.control_reason
        args, kwargs = captured[-1]
        np.testing.assert_allclose(args[4], rotation, atol=1e-14)
        expected_drag = compute_drag_force_e(vu, rotation, coordinator.drag_coefficients_b)
        np.testing.assert_allclose(kwargs['drag_force_e'], expected_drag, atol=1e-14)
        assert 'target_acceleration_e' not in kwargs
        assert np.isfinite(coordinator.inner_result.omega_d_b).all()
        invalid = ControllerTruth()
        coordinator._truth_callback(invalid)
        coordinator._update_control(coordinator.get_clock().now().nanoseconds)
        assert coordinator.observer_state is None and coordinator.px4_command is None
        subscriptions = {subscription.topic_name for subscription in coordinator.subscriptions}
        assert '/interception/controller_test/state' in subscriptions
        assert '/interception/observer/state' not in subscriptions
    finally:
        executor.remove_node(truth_node)
        executor.remove_node(coordinator)
        executor.shutdown()
        coordinator.destroy_node()


def test_test_launch_isolates_truth_topic_and_preserves_controller_parameters():
    path = Path(__file__).parents[1] / 'launch' / 'controller_truth_test.launch.py'
    spec = spec_from_file_location('controller_test_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    nodes = {action.node_executable: action for action in description.entities
             if isinstance(action, Node)}
    assert set(nodes) == {'controller_truth_state', 'controller_truth_coordinator'}
    truth, = evaluate_parameters(context, nodes['controller_truth_state']._Node__parameters)
    controller = {}
    for value in evaluate_parameters(
            context, nodes['controller_truth_coordinator']._Node__parameters):
        if isinstance(value, dict):
            controller.update(value)
        else:
            config = yaml.safe_load(Path(value).read_text())
            controller.update(config['vision_interception_coordinator']['ros__parameters'])
    assert truth['state_topic'] == controller['observer_topic']
    assert controller['observer_topic'] == '/interception/controller_test/state'
    assert not controller['enable_flight_commands']
    assert controller['timer_period_s'] == 0.005 and controller['outer_loop_hz'] == 50.0
    assert truth['stop_motion_on_shutdown']
    assert truth['camera_to_body_rotation'] == controller['camera_to_body_rotation']
    assert truth['camera_position_b_m'] == controller['camera_position_b_m']
    design_path = Path(__file__).parents[1] / 'config' / 'design_parameters.yaml'
    design = yaml.safe_load(design_path.read_text())
    for name, value in design['vision_interception_coordinator']['ros__parameters'].items():
        assert controller[name] == value
    bag, = [action for action in description.entities
            if isinstance(action, ExecuteProcess) and not isinstance(action, Node)]
    command = [perform_substitutions(context, normalize_to_list_of_substitutions(part))
               for part in bag.cmd]
    assert '/interception/controller_test/state' in command
    assert '/interception/control/debug' in command
    assert '/fmu/in/vehicle_rates_setpoint' in command
    assert '/fmu/out/vehicle_odometry' in command
    output = Path.home() / 'ibvs_ws/results/controller_truth/bags/controller_truth_01'
    assert str(output) in command


@pytest.mark.parametrize('moving', ('false', 'true'))
def test_native_probe_uses_the_test_library_and_requested_trajectory(moving):
    path = Path(__file__).parents[1] / 'launch' / 'controller_truth_test.launch.py'
    spec = spec_from_file_location('controller_probe_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations['moving_target'] = moving
    probe, = module.spawn_physics_probe(context)
    assert probe.node_executable == 'create'
    command = (perform_substitutions(context, normalize_to_list_of_substitutions(part))
               for part in probe.cmd)
    next(argument for argument in command if argument == '-string')
    sdf = ET.fromstring(next(command))
    plugin = sdf.find('model/plugin')
    assert Path(plugin.get('filename')).is_file()
    assert plugin.get('name') == 'ibvs_controller_test::ControllerTruthSystem'
    assert plugin.findtext('moving_target') == moving
    assert plugin.findtext('topic') == '/world/default/controller_test/state'
    assert plugin.findtext('target_pattern') == 'circle'
    assert float(plugin.findtext('target_speed_m_s')) == 4.0
    assert float(plugin.findtext('radius_x_m')) == 6.0


@pytest.mark.parametrize('filename', [
    'vision_direct_interception.launch.py',
    'vision_direct_interception_trial.launch.py',
])
def test_normal_launches_use_the_same_global_axis_gains(filename):
    path = Path(__file__).parents[1] / 'launch' / filename
    spec = spec_from_file_location('normal_control_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    coordinator, = [action for action in description.entities if isinstance(action, Node)
                    and action.node_executable == 'vision_interception_coordinator']
    controller = {}
    for value in evaluate_parameters(context, coordinator._Node__parameters):
        if isinstance(value, dict):
            controller.update(value)
        else:
            config = yaml.safe_load(Path(value).read_text())
            controller.update(config['vision_interception_coordinator']['ros__parameters'])
    design_path = Path(__file__).parents[1] / 'config' / 'design_parameters.yaml'
    design = yaml.safe_load(design_path.read_text())
    gains = design['vision_interception_coordinator']['ros__parameters']
    for name in ('velocity_gain', 'position_gain'):
        assert controller[name] == gains[name]
    assert controller['observer_topic'] == '/interception/observer/state'
