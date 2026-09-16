"""Spawn the static P2 target and publish truth relative state."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the non-controlling P2-A truth-monitor launch."""
    target_model = os.path.join(
        get_package_share_directory('ibvs_sim'),
        'models',
        'static_target.sdf',
    )
    world = LaunchConfiguration('world')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    target_z = LaunchConfiguration('target_z')

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )
    target_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='target_odometry_bridge',
        arguments=[
            '/model/ibvs_target/odometry@nav_msgs/msg/Odometry'
            '[gz.msgs.Odometry',
        ],
        output='screen',
    )
    contact_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='target_contact_bridge',
        arguments=[
            '/world/default/model/ibvs_target/link/target_link/sensor/'
            'target_contact/contact@ros_gz_interfaces/msg/Contacts'
            '[gz.msgs.Contacts',
        ],
        output='screen',
    )
    spawn_target = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_ibvs_target',
        arguments=[
            '-world', world,
            '-file', target_model,
            '-name', 'ibvs_target',
            '-x', target_x,
            '-y', target_y,
            '-z', target_z,
        ],
        output='screen',
    )
    truth_state = Node(
        package='ibvs_control',
        executable='truth_state',
        name='truth_state',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    contact_indicator = Node(
        package='ibvs_sim',
        executable='target_contact_indicator',
        name='target_contact_indicator',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='default'),
        # Place the first contact-verification target on the takeoff heading.
        # Cross-track tracking is a separate, later robustness scenario.
        DeclareLaunchArgument('target_x', default_value='0.0'),
        DeclareLaunchArgument('target_y', default_value='12.0'),
        DeclareLaunchArgument('target_z', default_value='3.0'),
        clock_bridge,
        target_bridge,
        contact_bridge,
        spawn_target,
        truth_state,
        contact_indicator,
    ])
