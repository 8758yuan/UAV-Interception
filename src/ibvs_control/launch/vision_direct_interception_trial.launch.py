"""Run one auditable delayed-DKF visual-interception trial."""

import os
from pathlib import Path
import re
import shutil

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
from launch_ros.parameter_descriptions import ParameterValue


def _prepare_directories(context) -> list:
    """Refuse unsafe ids, replacement, and recording onto a full disk."""
    trial_id = LaunchConfiguration('trial_id').perform(context)
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', trial_id) is None:
        raise ValueError('trial_id is not a safe 1-80 character identifier')
    results = Path(
        LaunchConfiguration('results_directory').perform(context)
    ).expanduser().resolve()
    if (results / 'bags' / trial_id).exists():
        raise FileExistsError(f'trial bag already exists: {trial_id}')
    (results / 'trials').mkdir(parents=True, exist_ok=True)
    (results / 'bags').mkdir(parents=True, exist_ok=True)
    record_bag = LaunchConfiguration('record_bag').perform(context)
    recording_enabled = record_bag.strip().lower() in ('1', 'true', 'yes', 'on')
    if recording_enabled:
        minimum_gb = float(
            LaunchConfiguration('minimum_bag_free_space_gb').perform(context)
        )
        if minimum_gb <= 0.0:
            raise ValueError('minimum_bag_free_space_gb must be positive')
        free_bytes = shutil.disk_usage(results).free
        required_bytes = int(minimum_gb * 1024**3)
        if free_bytes < required_bytes:
            free_gb = free_bytes / 1024**3
            raise RuntimeError(
                'insufficient disk space for trial recording: '
                f'{free_gb:.2f} GiB free, {minimum_gb:.2f} GiB required'
            )
    return []


def generate_launch_description() -> LaunchDescription:
    """Build a bounded trial with command output disabled by default."""
    package_share = get_package_share_directory('ibvs_control')
    config = os.path.join(
        package_share, 'config', 'vision_direct_interception.yaml'
    )
    observer_config = os.path.join(
        package_share, 'config', 'paper_state_observer.yaml'
    )
    paper_design_config = os.path.join(
        package_share, 'config', 'paper_design_parameters.yaml'
    )
    trial_id = LaunchConfiguration('trial_id')
    results = LaunchConfiguration('results_directory')
    enabled = LaunchConfiguration('enable_flight_commands')
    token = LaunchConfiguration('confirmation_token')
    record_bag = LaunchConfiguration('record_bag')
    speed_limit = LaunchConfiguration('speed_limit_m_s')
    tilt_limit = LaunchConfiguration('interception_tilt_limit_deg')
    horizontal_limit = LaunchConfiguration('max_horizontal_distance_m')
    coordinator = Node(
        package='ibvs_control',
        executable='vision_interception_coordinator',
        name='vision_interception_coordinator',
        output='screen',
        parameters=[
            config,
            paper_design_config,
            {
                'enable_flight_commands': ParameterValue(
                    enabled,
                    value_type=bool,
                ),
                'confirmation_token': token,
                'speed_limit_m_s': ParameterValue(
                    speed_limit,
                    value_type=float,
                ),
                'interception_tilt_limit_deg': ParameterValue(
                    tilt_limit,
                    value_type=float,
                ),
                'max_horizontal_distance_m': ParameterValue(
                    horizontal_limit,
                    value_type=float,
                ),
            },
        ],
    )
    observer = Node(
        package='ibvs_control',
        executable='paper_state_observer',
        name='paper_state_observer',
        output='screen',
        parameters=[observer_config, paper_design_config],
    )
    bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-o',
            PathJoinSubstitution([results, 'bags', trial_id]),
            '/clock',
            # Raw 1280x960 frames consumed about 110 MB/s and filled the disk.
            # VisionFeature plus CameraInfo retain the controller evidence.
            '/camera/camera_info',
            '/interception/vision/raw_feature',
            '/interception/observer/state',
            '/interception/observer/reset',
            '/interception/control/debug',
            '/interception/target/green_confirmed',
            '/world/default/model/ibvs_target/link/target_link/sensor/'
            'target_contact/contact',
            '/fmu/in/offboard_control_mode',
            '/fmu/in/trajectory_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
            '/fmu/in/vehicle_command',
            '/fmu/out/vehicle_attitude',
            '/fmu/out/sensor_combined',
            '/fmu/out/vehicle_land_detected',
            '/fmu/out/vehicle_local_position_v1',
            '/fmu/out/vehicle_odometry',
            '/fmu/out/vehicle_status_v4',
        ],
        output='screen',
        condition=IfCondition(record_bag),
    )
    shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=coordinator,
            on_exit=[
                EmitEvent(event=Shutdown(reason='P4 trial reached a safe result'))
            ],
        )
    )
    default_results = os.path.join(
        os.path.expanduser('~'), 'ibvs_ws', 'results', 'p4'
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument('trial_id', default_value='p4_vision_01'),
            DeclareLaunchArgument(
                'results_directory', default_value=default_results
            ),
            DeclareLaunchArgument(
                'enable_flight_commands', default_value='false'
            ),
            DeclareLaunchArgument('confirmation_token', default_value=''),
            DeclareLaunchArgument('record_bag', default_value='true'),
            DeclareLaunchArgument(
                'minimum_bag_free_space_gb',
                default_value='1.0',
                description='启动录包所需的最小剩余磁盘空间，单位GiB',
            ),
            DeclareLaunchArgument(
                'speed_limit_m_s',
                default_value='21.0',
                description='拦截阶段最大飞行速度，单位m/s',
            ),
            DeclareLaunchArgument(
                'interception_tilt_limit_deg',
                default_value='55.0',
                description='拦截阶段最大倾角，单位deg',
            ),
            DeclareLaunchArgument(
                'max_horizontal_distance_m',
                default_value='20.0',
                description='相对起飞点的最大水平距离，单位m',
            ),
            OpaqueFunction(function=_prepare_directories),
            shutdown,
            bag,
            observer,
            coordinator,
        ]
    )
