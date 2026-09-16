"""Launch the read-only P2 SO(3) controller shadow."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Load frozen offline parameters and start the shadow node."""
    parameters = os.path.join(
        get_package_share_directory('ibvs_control'),
        'config',
        'controller_shadow.yaml',
    )
    return LaunchDescription([
        Node(
            package='ibvs_control',
            executable='controller_shadow',
            name='controller_shadow',
            parameters=[parameters],
            output='screen',
        ),
    ])
