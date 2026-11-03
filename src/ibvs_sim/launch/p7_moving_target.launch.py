"""Launch the paper moving-target scenario without target truth topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


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
            'moving_target': 'true',
            'target_pattern': pattern,
            'target_speed_m_s': target_speed,
            'figure8_radius_x_m': radius_x,
            'figure8_radius_y_m': radius_y,
            'target_tilt_deg': LaunchConfiguration('target_tilt_deg'),
            'target_x': target_x,
            'target_y': target_y,
            'target_z': target_z,
            'vehicle_model_name': model_name,
            'device': LaunchConfiguration('device'),
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='default'),
        DeclareLaunchArgument('target_x', default_value='12.0'),
        DeclareLaunchArgument('target_y', default_value='0.0'),
        DeclareLaunchArgument('target_z', default_value='4.0'),
        DeclareLaunchArgument('target_pattern', default_value='circle'),
        DeclareLaunchArgument('target_tilt_deg', default_value='45.0'),
        DeclareLaunchArgument('target_speed_m_s', default_value='5.0'),
        DeclareLaunchArgument('figure8_radius_x_m', default_value='6.0'),
        DeclareLaunchArgument('figure8_radius_y_m', default_value='6.0'),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument(
            'vehicle_model_name', default_value='x500_mono_cam_0'
        ),
        vision_and_target,
    ])
