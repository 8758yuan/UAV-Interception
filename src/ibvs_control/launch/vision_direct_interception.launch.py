"""Run the hard-disabled camera-feature-only interception coordinator."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
            'paper_design_parameters.yaml',
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'approach_profile_enabled', default_value='true'
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
                parameters=[
                    config, paper_design_config,
                    {'approach_profile.enabled': ParameterValue(
                        LaunchConfiguration('approach_profile_enabled'),
                        value_type=bool,
                    )},
                ],
            )
        ]
    )
