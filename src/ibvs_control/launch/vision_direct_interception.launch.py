"""Run the hard-disabled camera-feature-only interception coordinator."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Load vision-only configuration; it remains debug-only by default."""
    config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'vision_direct_interception.yaml',
        ]
    )
    return LaunchDescription(
        [
            Node(
                package='ibvs_control',
                executable='vision_interception_coordinator',
                name='vision_interception_coordinator',
                output='screen',
                parameters=[config],
            )
        ]
    )
