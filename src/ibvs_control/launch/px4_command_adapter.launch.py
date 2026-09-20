"""Launch the PX4 adapter with its command output disabled by default."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Load the reviewed, disabled adapter configuration."""
    config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'px4_command_adapter.yaml',
        ]
    )
    return LaunchDescription(
        [
            Node(
                package='ibvs_control',
                executable='px4_command_adapter',
                name='px4_command_adapter',
                output='screen',
                parameters=[config],
            )
        ]
    )
