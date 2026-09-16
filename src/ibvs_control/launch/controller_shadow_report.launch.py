"""Launch controller shadow plus a bounded JSON metrics recorder."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Start read-only control calculation and its report recorder."""
    config = PathJoinSubstitution(
        [FindPackageShare('ibvs_control'), 'config', 'controller_shadow.yaml']
    )
    duration = LaunchConfiguration('duration_s')
    trial_id = LaunchConfiguration('trial_id')
    output_directory = LaunchConfiguration('output_directory')
    return LaunchDescription(
        [
            DeclareLaunchArgument('duration_s', default_value='30.0'),
            DeclareLaunchArgument('trial_id', default_value='p2_shadow_01'),
            DeclareLaunchArgument(
                'output_directory',
                default_value='results/p2/shadow',
            ),
            Node(
                package='ibvs_control',
                executable='controller_shadow',
                name='controller_shadow',
                output='screen',
                parameters=[config],
            ),
            Node(
                package='ibvs_control',
                executable='controller_shadow_report',
                name='controller_shadow_report',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'safe_los_angle_deg': 45.0,
                        'require_airborne': True,
                        'airborne_settle_s': 12.0,
                        'duration_s': duration,
                        'trial_id': trial_id,
                        'output_directory': output_directory,
                    },
                ],
            ),
        ]
    )
