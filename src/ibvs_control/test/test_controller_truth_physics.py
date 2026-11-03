"""Headless physics verification, isolated from PX4 and all normal Gazebo worlds."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import pytest


def _run_physics_check(pattern, with_probe):
    """Compare the exported snapshot with independent native component reads."""
    os.environ['GZ_PARTITION'] = f'ibvs-physics-test-{os.getpid()}'
    from ament_index_python.packages import get_package_prefix
    from importlib.util import module_from_spec, spec_from_file_location
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
    from gz.math7 import Vector3d
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.param_v_pb2 import Param_V
    from gz.sim8 import Link, Model, TestFixture, World, world_entity
    from gz.transport13 import Node
    import numpy as np

    from ibvs_sim.paper_moving_target import PaperTargetTrajectory

    root = ET.Element('sdf', version='1.9')
    world = ET.SubElement(root, 'world', name='native_check')
    ET.SubElement(world, 'gravity').text = '0 0 0'
    physics = ET.SubElement(world, 'physics', name='p', type='ignored')
    ET.SubElement(physics, 'max_step_size').text = '0.004'
    ET.SubElement(physics, 'real_time_factor').text = '0'
    ET.SubElement(world, 'plugin', filename='gz-sim-physics-system',
                  name='gz::sim::systems::Physics')
    vehicle = ET.SubElement(world, 'model', name='x500_mono_cam_0')
    ET.SubElement(vehicle, 'pose').text = '1 -2 3 0.2 0.1 0.3'
    link = ET.SubElement(vehicle, 'link', name='base_link')
    ET.SubElement(link, 'gravity').text = 'false'
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass').text = '2'
    inertia = ET.SubElement(inertial, 'inertia')
    for name in ('ixx', 'iyy', 'izz'):
        ET.SubElement(inertia, name).text = '0.1'
    # Use the actual scenario's spawn path, shared by visual, DKF and truth tests.
    launch_path = Path(__file__).parents[2] / 'ibvs_sim/launch/p4_vision_monitor.launch.py'
    spec = spec_from_file_location('target_launch', launch_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations.update({
        'moving_target': 'false' if pattern == 'static' else 'true',
        'target_pattern': pattern, 'target_speed_m_s': '4.0',
    })
    spawn, = module.spawn_target(context)
    command = (perform_substitutions(context, normalize_to_list_of_substitutions(part))
               for part in spawn.cmd)
    next(argument for argument in command if argument == '-string')
    target = ET.fromstring(next(command)).find('model')
    ET.SubElement(target, 'pose').text = '35 0 14 0.2 0.1 0.7'
    world.append(target)
    topic = '/world/native_check/controller_test/state'
    if with_probe:
        probe = ET.SubElement(world, 'model', name='controller_truth_probe')
        ET.SubElement(probe, 'static').text = 'true'
        ET.SubElement(probe, 'link', name='probe')
        library = (Path(get_package_prefix('ibvs_controller_test'))
                   / 'lib/libcontroller_truth_system.so')
        plugin = ET.SubElement(probe, 'plugin', filename=str(library),
                               name='ibvs_controller_test::ControllerTruthSystem')
        for name, value in dict(vehicle_model='x500_mono_cam_0',
                                target_model='ibvs_target', topic=topic).items():
            ET.SubElement(plugin, name).text = value
    received, expected, links = {}, {}, {}
    event = threading.Event()
    force_target = False
    node = Node()

    def receive(message):
        stamp = message.header.stamp
        received[stamp.sec * 10**9 + stamp.nsec] = message
        event.set()

    assert node.subscribe(Param_V, topic, receive)

    def pre(_info, ecm):
        if not links:
            native_world = World(world_entity(ecm))
            for name, model, link_name in (
                ('vehicle', 'x500_mono_cam_0', 'base_link'),
                ('target', 'ibvs_target', 'target_link'),
            ):
                model_id = native_world.model_by_name(ecm, model)
                links[name] = Link(Model(model_id).link_by_name(ecm, link_name))
                links[name].enable_velocity_checks(ecm)
        links['vehicle'].add_world_force(ecm, Vector3d(0.2, 0.4, -0.1))
        if force_target:
            links['target'].add_world_force(ecm, Vector3d(0.02, 0., 0.))

    def vector(value):
        return np.array([value.x(), value.y(), value.z()])

    def post(info, ecm):
        stamp = ((info.sim_time.days * 86400 + info.sim_time.seconds) * 10**9
                 + info.sim_time.microseconds * 1000)
        fields = {}
        for name, native_link in links.items():
            pose = native_link.world_pose(ecm)
            fields[name + '_position'] = vector(pose.pos())
            fields[name + '_q'] = np.array([pose.rot().w(), pose.rot().x(),
                                           pose.rot().y(), pose.rot().z()])
            fields[name + '_velocity'] = vector(native_link.world_linear_velocity(ecm))
            fields[name + '_angular'] = vector(native_link.world_angular_velocity(ecm))
        expected[stamp] = fields

    with tempfile.TemporaryDirectory(prefix='ibvs-native-physics-') as directory:
        sdf_path = Path(directory) / 'world.sdf'
        ET.ElementTree(root).write(sdf_path, encoding='unicode')
        fixture = TestFixture(str(sdf_path))
        fixture.on_pre_update(pre)
        fixture.on_post_update(post)
        fixture.finalize()
        # Allow Gazebo transport discovery before running faster than real time.
        time.sleep(0.2)
        assert fixture.server().run(True, 4000, False)
        deadline = time.monotonic() + 3.0
        while with_probe and (not received or max(received) < max(expected)) and time.monotonic() < deadline:
            event.wait(0.02)
            event.clear()
        if with_probe:
            assert len(received) > 10 and max(received) == max(expected)
        trajectory = PaperTargetTrajectory('circle' if pattern == 'static' else pattern,
                                           (35., 0., 14.), 4., 6., 6., tilt_deg=45.)
        for stamp, message in received.items():
            fields = message.param[0].params
            native = expected[stamp]
            for name in ('vehicle', 'target'):
                pose = fields[name + '_pose'].pose3d_value
                velocity = fields[name + '_velocity'].vector3d_value
                np.testing.assert_array_equal(
                    [pose.position.x, pose.position.y, pose.position.z],
                    native[name + '_position'],
                )
                np.testing.assert_array_equal(
                    [pose.orientation.w, pose.orientation.x,
                     pose.orientation.y, pose.orientation.z],
                    native[name + '_q'],
                )
                np.testing.assert_array_equal(
                    [velocity.x, velocity.y, velocity.z], native[name + '_velocity'],
                )
            angular = fields['vehicle_angular_velocity'].vector3d_value
            np.testing.assert_array_equal([angular.x, angular.y, angular.z],
                                          native['vehicle_angular'])
        previous_position = np.array([35., 0., 14.])
        previous_stamp = 0
        for stamp, native in expected.items():
            elapsed = stamp * 1e-9
            if pattern == 'static':
                target_position, target_velocity = np.array([35., 0., 14.]), np.zeros(3)
            else:
                target_position = np.array(trajectory.sample(elapsed).position)
                previous = np.array(trajectory.sample(max(elapsed - 0.004, 0.)).position)
                target_velocity = (target_position - previous) / 0.004
            np.testing.assert_allclose(native['target_position'], target_position,
                                       atol=1e-7, rtol=0)
            np.testing.assert_allclose(native['target_velocity'], target_velocity,
                                       atol=1e-8, rtol=0)
            measured_velocity = ((native['target_position'] - previous_position)
                                 / ((stamp - previous_stamp) * 1e-9))
            np.testing.assert_allclose(native['target_velocity'], measured_velocity,
                                       atol=1e-8, rtol=0)
            previous_position, previous_stamp = native['target_position'], stamp
            np.testing.assert_allclose(native['vehicle_velocity'],
                                       np.array([0.1, 0.2, -0.05]) * elapsed, atol=1e-12)
        if pattern != 'static':
            executed, response = node.request('/world/native_check/paper_target/stop',
                                              Empty(), Empty, Boolean, 1000)
            assert executed and response.data
        before = expected[max(expected)]
        force_target = True
        assert fixture.server().run(True, 10, False)
        after = expected[max(expected)]
        # A released target responds to force rather than a retained velocity command.
        np.testing.assert_allclose(after['target_velocity'] - before['target_velocity'],
                                   [0.04, 0., 0.], atol=1e-10)
        node.unsubscribe(topic)


@pytest.mark.parametrize('pattern', ('static', 'circle', 'figure8'))
@pytest.mark.parametrize('with_probe', (False, True))
def test_same_step_native_snapshot_and_physical_target_motion(pattern, with_probe):
    result = subprocess.run([sys.executable, '-s', str(Path(__file__).resolve()),
                             pattern, str(int(with_probe))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    _run_physics_check(sys.argv[1], bool(int(sys.argv[2])))
