"""Spawn the target and start the undelayed forward-camera data path."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Compose image features with an isolated evaluation sidecar."""
    world = LaunchConfiguration('world')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    target_z = LaunchConfiguration('target_z')
    model_name = LaunchConfiguration('vehicle_model_name')
    camera_image_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/image',
    ]
    camera_info_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/camera_info',
    ]
    truth_monitor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ibvs_sim'),
                    'launch',
                    'p2_truth_monitor.launch.py',
                ]
            )
        ),
        launch_arguments={
            'world': world,
            'target_x': target_x,
            'target_y': target_y,
            'target_z': target_z,
        }.items(),
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
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='default'),
            # mono_cam points along Gazebo +X for the x500_mono_cam model.
            DeclareLaunchArgument('target_x', default_value='12.0'),
            DeclareLaunchArgument('target_y', default_value='0.0'),
            DeclareLaunchArgument('target_z', default_value='3.0'),
            DeclareLaunchArgument(
                'vehicle_model_name', default_value='x500_mono_cam_0'
            ),
            truth_monitor,
            camera_bridge,
            detector,
        ]
    )
