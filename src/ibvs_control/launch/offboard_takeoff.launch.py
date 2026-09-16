"""Launch the P0 PX4 Offboard takeoff node with its default parameters."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import os


def generate_launch_description() -> LaunchDescription:
    """Build the Offboard takeoff launch description."""
    package_share = get_package_share_directory('ibvs_control')
    parameters_file = os.path.join(
        package_share,
        'config',
        'offboard_takeoff.yaml',
    )

    return LaunchDescription([
        Node(
            package='ibvs_control',
            executable='offboard_takeoff',
            name='offboard_takeoff',
            output='screen',
            parameters=[parameters_file],
        ),
    ])
