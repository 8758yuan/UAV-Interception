"""Launch the paper's 18-state observer with delayed-image DKF updates."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Load the observer parameters and start its ROS node."""
    config = PathJoinSubstitution(
        [
            FindPackageShare('ibvs_control'),
            'config',
            'paper_state_observer.yaml',
        ]
    )
    return LaunchDescription(
        [
            Node(
                package='ibvs_control',
                executable='paper_state_observer',
                name='paper_state_observer',
                output='screen',
                parameters=[config],
            )
        ]
    )
