"""Spawn the target and start YOLO detection on the forward camera."""

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def spawn_target(context):
    """Attach the sole physical motion driver when spawning a moving target."""
    def value(name):
        return LaunchConfiguration(name).perform(context)

    path = os.path.join(get_package_share_directory('ibvs_sim'),
                        'target_models', 'static_target.sdf')
    sdf = ET.parse(path).getroot()
    if value('moving_target').lower() == 'true':
        library = os.path.join(get_package_prefix('ibvs_sim_physics'),
                               'lib', 'libpaper_target_system.so')
        plugin = ET.SubElement(sdf.find('model'), 'plugin', filename=library,
                               name='ibvs_sim_physics::PaperTargetSystem')
        parameters = {
            'vehicle_model': value('vehicle_model_name'),
            'target_pattern': value('target_pattern'),
            'target_speed_m_s': value('target_speed_m_s'),
            'radius_x_m': value('figure8_radius_x_m'),
            'radius_y_m': value('figure8_radius_y_m'),
            'target_tilt_deg': value('target_tilt_deg'),
        }
        for name, parameter in parameters.items():
            ET.SubElement(plugin, name).text = parameter
    return [Node(
        package='ros_gz_sim', executable='create', name='spawn_ibvs_target',
        arguments=['-world', value('world'), '-string', ET.tostring(sdf, encoding='unicode'),
                   '-name', 'ibvs_target', '-x', value('target_x'),
                   '-y', value('target_y'), '-z', value('target_z')],
        output='screen',
    )]


def stop_target(context):
    """Release the trajectory command when the scenario launch exits."""
    if LaunchConfiguration('moving_target').perform(context).lower() == 'true':
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.empty_pb2 import Empty
        from gz.transport13 import Node as GazeboNode

        world = LaunchConfiguration('world').perform(context)
        GazeboNode().request(f'/world/{world}/paper_target/stop', Empty(), Empty, Boolean, 100)
    return []


def generate_launch_description() -> LaunchDescription:
    """Compose target contact feedback and image features."""
    world = LaunchConfiguration('world')
    model_name = LaunchConfiguration('vehicle_model_name')
    image_delay = LaunchConfiguration('image_delay_s')
    model_path = LaunchConfiguration('model_path')
    target_class = LaunchConfiguration('target_class')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    inference_size = LaunchConfiguration('inference_size')
    device = LaunchConfiguration('device')
    camera_image_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/image',
    ]
    camera_info_gz = [
        '/world/', world, '/model/', model_name,
        '/link/camera_link/sensor/camera/camera_info',
    ]
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
            camera_info_gz
            + ['@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
        ],
        remappings=[
            (camera_image_gz, '/camera/image_raw'),
            (camera_info_gz, '/camera/camera_info'),
        ],
        output='screen',
    )
    detector = Node(
        package='ibvs_perception',
        executable='yolo_target_detector',
        name='yolo_target_detector',
        parameters=[{
            'use_sim_time': True,
            'image_delay_s': ParameterValue(image_delay, value_type=float),
            'model_path': model_path,
            'target_class': target_class,
            'confidence_threshold': ParameterValue(
                confidence_threshold, value_type=float
            ),
            'inference_size': ParameterValue(inference_size, value_type=int),
            'device': ParameterValue(device, value_type=str),
        }],
        output='screen',
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='default'),
            # mono_cam points along Gazebo +X for the x500_mono_cam model.
            DeclareLaunchArgument('target_x', default_value='12.0'),
            DeclareLaunchArgument('target_y', default_value='0.0'),
            # Keep the nominal regression target on the initial search plane.
            # Other heights are supported by the coordinator's visual
            # acquisition scan, so this value is intentionally independent of
            # the controller's initial takeoff altitude.
            DeclareLaunchArgument('target_z', default_value='4.0'),
            DeclareLaunchArgument('moving_target', default_value='false'),
            DeclareLaunchArgument('target_pattern', default_value='circle'),
            DeclareLaunchArgument('target_speed_m_s', default_value='5.0'),
            DeclareLaunchArgument('figure8_radius_x_m', default_value='6.0'),
            DeclareLaunchArgument('figure8_radius_y_m', default_value='6.0'),
            DeclareLaunchArgument('target_tilt_deg', default_value='45.0'),
            DeclareLaunchArgument(
                'vehicle_model_name', default_value='x500_mono_cam_0'
            ),
            # SITL capture-to-publication delay is configured to 70 ms.
            DeclareLaunchArgument('image_delay_s', default_value='0.07'),
            DeclareLaunchArgument(
                'model_path',
                default_value=os.path.join(
                    get_package_share_directory('ibvs_perception'),
                    'models', 'balloon_yolo26n.pt',
                ),
                description=(
                    'YOLO26n balloon weights; custom weights are supported'
                ),
            ),
            DeclareLaunchArgument(
                'target_class', default_value='balloon',
                description='Exact class name in the selected YOLO weights',
            ),
            DeclareLaunchArgument(
                'confidence_threshold', default_value='0.25'
            ),
            DeclareLaunchArgument('inference_size', default_value='1280'),
            DeclareLaunchArgument('device', default_value=''),
            clock_bridge,
            contact_bridge,
            OpaqueFunction(function=spawn_target),
            RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=stop_target)])),
            contact_indicator,
            camera_bridge,
            detector,
        ]
    )
