"""Launch independent A/B/C/D trials and reuse the DKF half-second reports."""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _launch_test(context):
    test_case = LaunchConfiguration('test_case').perform(context).upper()
    if test_case not in ('A', 'B', 'C', 'D'):
        raise ValueError('test_case must be A, B, C or D')
    control_share = get_package_share_directory('ibvs_control')

    def parameters(filename, node):
        with open(os.path.join(control_share, 'config', filename), encoding='utf-8') as stream:
            return yaml.safe_load(stream)[node]['ros__parameters']

    observer = parameters('paper_state_observer.yaml', 'paper_state_observer')
    # Reuse current size-bootstrap settings without changing the flight YAML.
    flight = parameters('vision_direct_interception.yaml', 'vision_interception_coordinator')
    flight.update(parameters('design_parameters.yaml', 'vision_interception_coordinator'))
    takeoff = parameters('offboard_takeoff.yaml', 'offboard_takeoff')
    flight.update(takeoff)
    flight.update({
        'enable_flight_commands': True,
        'confirmation_token': 'ENABLE_VISION_ONLY_CONTROL',
        'test_case': test_case, 'timer_period_s': 0.02,
        'target_altitude_m': 10.0, 'target_yaw_rad': math.pi / 2.0,
        'debug_topic': '/interception/dkf_test/status',
        'camera_to_body_rotation': observer['camera_to_body_rotation'],
        'camera_position_b_m': observer['camera_position_b_m'],
    })
    runner = Node(
        package='ibvs_control', executable='dkf_standalone_test',
        name='dkf_standalone_test', output='screen', parameters=[flight],
    )
    return [
        RegisterEventHandler(OnProcessExit(
            target_action=runner,
            on_exit=[EmitEvent(event=Shutdown(reason='DKF standalone test finished'))],
        )),
        Node(
            package='ibvs_control', executable='dkf_evaluation',
            name='dkf_evaluation', output='screen',
            parameters=[{
                'use_sim_time': True, 'test_case': test_case,
                'world_name': LaunchConfiguration('world'),
                'vehicle_model': LaunchConfiguration('vehicle_model_name'),
                'output_directory': LaunchConfiguration('output_directory'),
                'run_id': LaunchConfiguration('trial_id'),
                'camera_to_body_rotation': observer['camera_to_body_rotation'],
                'camera_position_b_m': observer['camera_position_b_m'],
            }],
            remappings=[('/interception/control/debug', '/interception/dkf_test/status')],
        ),
        Node(
            package='ibvs_control', executable='paper_state_observer',
            name='paper_state_observer', output='screen',
            parameters=[
                os.path.join(control_share, 'config', 'paper_state_observer.yaml'),
                os.path.join(control_share, 'config', 'design_parameters.yaml'),
            ],
        ),
        runner,
    ]


def generate_launch_description():
    """Start control/measurement nodes; target and camera use their usual launch."""
    return LaunchDescription([
        DeclareLaunchArgument('test_case', default_value='C'),
        DeclareLaunchArgument('trial_id', default_value=''),
        DeclareLaunchArgument('world', default_value='default'),
        DeclareLaunchArgument('vehicle_model_name', default_value='x500_mono_cam_0'),
        DeclareLaunchArgument(
            'output_directory', default_value=os.path.expanduser('~/ibvs_ws/results/p4/dkf'),
        ),
        OpaqueFunction(function=_launch_test),
    ])
