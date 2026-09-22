"""Launch the paper moving-target scenario without target truth topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Spawn vision/contact nodes and a configurable paper target path."""
    world = LaunchConfiguration('world')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    target_z = LaunchConfiguration('target_z')
    target_speed = LaunchConfiguration('target_speed_m_s')
    radius_x = LaunchConfiguration('figure8_radius_x_m')
    radius_y = LaunchConfiguration('figure8_radius_y_m')
    pattern = LaunchConfiguration('target_pattern')
    model_name = LaunchConfiguration('vehicle_model_name')
    base_launch = os.path.join(
        get_package_share_directory('ibvs_sim'),
        'launch',
        'p4_vision_monitor.launch.py',
    )
    vision_and_target = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            'world': world,
            'target_x': target_x,
            'target_y': target_y,
            'target_z': target_z,
            'vehicle_model_name': model_name,
        }.items(),
    )
    mover = Node(
        package='ibvs_sim',
        executable='paper_moving_target',
        name='paper_moving_target',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'world_name': world,
            'model_name': 'ibvs_target',
            'pattern': pattern,
            'origin_x_m': ParameterValue(target_x, value_type=float),
            'origin_y_m': ParameterValue(target_y, value_type=float),
            'origin_z_m': ParameterValue(target_z, value_type=float),
            'speed_m_s': ParameterValue(target_speed, value_type=float),
            'radius_x_m': ParameterValue(radius_x, value_type=float),
            'radius_y_m': ParameterValue(radius_y, value_type=float),
            'update_rate_hz': 100.0,
            'start_on_observer_reset': True,
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='default'),
        DeclareLaunchArgument('target_x', default_value='12.0'),
        DeclareLaunchArgument('target_y', default_value='0.0'),
        # Keep the nominal moving-target centre on the initial search plane;
        # other heights are handled by visual acquisition.
        DeclareLaunchArgument('target_z', default_value='4.0'),
        DeclareLaunchArgument('target_pattern', default_value='figure8'),
        # Paper HITL evaluates the figure-eight at 5, 7.5, and 10 m/s.
        DeclareLaunchArgument('target_speed_m_s', default_value='5.0'),
        DeclareLaunchArgument('figure8_radius_x_m', default_value='4.0'),
        DeclareLaunchArgument('figure8_radius_y_m', default_value='2.0'),
        DeclareLaunchArgument(
            'vehicle_model_name', default_value='x500_mono_cam_0'
        ),
        vision_and_target,
        mover,
    ])
