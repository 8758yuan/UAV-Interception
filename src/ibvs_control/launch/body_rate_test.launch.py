"""Launch the P1 body-rate and thrust direction test."""

import os
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _prepare_output_directories(context) -> list:
    """Validate the trial identifier and create rosbag parent directories."""
    trial_id = LaunchConfiguration('trial_id').perform(context)
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', trial_id) is None:
        raise ValueError(
            'trial_id must use 1-80 letters, numbers, dot, dash, or underscore'
        )
    results_directory = LaunchConfiguration('results_directory').perform(
        context
    )
    if not results_directory.strip():
        raise ValueError('results_directory must not be empty')
    output_dir = Path(results_directory).expanduser().resolve()
    (output_dir / 'bags').mkdir(parents=True, exist_ok=True)
    return []


def generate_launch_description() -> LaunchDescription:
    """Build the body-rate test launch description."""
    package_share = get_package_share_directory('ibvs_control')
    parameters_file = os.path.join(
        package_share,
        'config',
        'body_rate_test.yaml',
    )
    default_results_directory = os.environ.get(
        'IBVS_RESULTS_DIR',
        os.path.join(os.path.expanduser('~'), 'ibvs_ws', 'results', 'p1'),
    )
    trial_id = LaunchConfiguration('trial_id')
    results_directory = LaunchConfiguration('results_directory')
    record_bag = LaunchConfiguration('record_bag')

    test_node = Node(
        package='ibvs_control',
        executable='body_rate_test',
        name='body_rate_test',
        output='screen',
        parameters=[
            parameters_file,
            {
                'trial_id': trial_id,
                'results_directory': results_directory,
            },
        ],
    )
    bag_output = PathJoinSubstitution([
        results_directory,
        'bags',
        trial_id,
    ])
    bag_recorder = ExecuteProcess(
        cmd=[
            'ros2',
            'bag',
            'record',
            '-o',
            bag_output,
            '/clock',
            '/fmu/in/offboard_control_mode',
            '/fmu/in/trajectory_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
            '/fmu/in/vehicle_command',
            '/fmu/out/vehicle_attitude',
            '/fmu/out/vehicle_command_ack_v1',
            '/fmu/out/vehicle_land_detected',
            '/fmu/out/vehicle_local_position_v1',
            '/fmu/out/vehicle_odometry',
            '/fmu/out/vehicle_status_v4',
        ],
        output='screen',
        condition=IfCondition(record_bag),
    )
    shutdown_after_test = RegisterEventHandler(
        OnProcessExit(
            target_action=test_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(
                        reason='P1 body-rate trial reached a terminal result',
                    ),
                ),
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'trial_id',
            default_value='manual',
            description='unique identifier for this P1 trial',
        ),
        DeclareLaunchArgument(
            'results_directory',
            default_value=default_results_directory,
            description='directory for P1 JSON, CSV, and bag results',
        ),
        DeclareLaunchArgument(
            'record_bag',
            default_value='false',
            description='record the P1 evidence topics when true',
        ),
        OpaqueFunction(function=_prepare_output_directories),
        shutdown_after_test,
        bag_recorder,
        test_node,
    ])
