"""Run the hard-disabled camera-feature-only interception coordinator."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Start observer and disabled-by-default visual control."""
    config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'vision_direct_interception.yaml',
        ]
    )
    observer_config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'paper_state_observer.yaml',
        ]
    )
    paper_design_config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'design_parameters.yaml',
        ]
    )
    observer_path = os.path.join(
        get_package_share_directory('ibvs_control'), 'config', 'paper_state_observer.yaml'
    )
    with open(observer_path, encoding='utf-8') as stream:
        observer_parameters = yaml.safe_load(stream)['paper_state_observer']['ros__parameters']
    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='default'),
            DeclareLaunchArgument('vehicle_model_name', default_value='x500_mono_cam_0'),
            Node(
                package='ibvs_control',
                executable='dkf_evaluation',
                name='dkf_evaluation',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'world_name': LaunchConfiguration('world'),
                    'vehicle_model': LaunchConfiguration('vehicle_model_name'),
                    'camera_to_body_rotation': observer_parameters['camera_to_body_rotation'],
                    'camera_position_b_m': observer_parameters['camera_position_b_m'],
                }],
            ),
            Node(
                package='ibvs_control',
                executable='paper_state_observer',
                name='paper_state_observer',
                output='screen',
                parameters=[observer_config, paper_design_config],
            ),
            Node(
                package='ibvs_control',
                executable='vision_interception_coordinator',
                name='vision_interception_coordinator',
                output='screen',
                parameters=[config, paper_design_config],
            )
        ]
    )
