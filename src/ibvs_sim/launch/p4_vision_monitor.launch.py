"""Spawn the target and start the delayed forward-camera data path."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Compose target contact feedback and image features without world state."""
    world = LaunchConfiguration('world')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    target_z = LaunchConfiguration('target_z')
    model_name = LaunchConfiguration('vehicle_model_name')
    image_delay = LaunchConfiguration('image_delay_s')
    camera_image_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/image',
    ]
    camera_info_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/camera_info',
    ]
    target_model = os.path.join(
        get_package_share_directory('ibvs_sim'),
        'models',
        'static_target.sdf',
    )
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
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
    contact_indicator = Node(
        package='ibvs_sim',
        executable='target_contact_indicator',
        name='target_contact_indicator',
        output='screen',
    )
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='front_camera_bridge',
        arguments=[
            camera_image_gz + ['@sensor_msgs/msg/Image[gz.msgs.Image'],
            camera_info_gz + ['@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
        ],
        remappings=[
            (camera_image_gz, '/camera/image_raw'),
            (camera_info_gz, '/camera/camera_info'),
        ],
        output='screen',
    )
    detector = Node(
        package='ibvs_perception',
        executable='red_target_detector',
        name='red_target_detector',
        parameters=[{
            'use_sim_time': True,
            'image_delay_s': ParameterValue(image_delay, value_type=float),
        }],
        output='screen',
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='default'),
            # mono_cam points along Gazebo +X for the x500_mono_cam model.
            DeclareLaunchArgument('target_x', default_value='12.0'),
            DeclareLaunchArgument('target_y', default_value='0.0'),
            # The x500_mono_cam optical centre is at the vehicle body height;
            # vision_direct_interception targets 4 m, so keep the balloon
            # centre on that same horizontal plane by default.
            DeclareLaunchArgument('target_z', default_value='4.0'),
            DeclareLaunchArgument(
                'vehicle_model_name', default_value='x500_mono_cam_0'
            ),
            # Paper flight experiments report about 80 ms total imaging and
            # processing delay.  This makes SITL exercise the same DKF path.
            DeclareLaunchArgument('image_delay_s', default_value='0.08'),
            clock_bridge,
            contact_bridge,
            spawn_target,
            contact_indicator,
            camera_bridge,
            detector,
        ]
    )
