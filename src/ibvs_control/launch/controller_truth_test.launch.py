"""Run the existing controller against Gazebo truth without starting the DKF."""

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def spawn_physics_probe(context):
    """Load the native exporter and optional target driver only in this test."""
    def value(name):
        return LaunchConfiguration(name).perform(context)

    world = value('world')
    sdf = ET.Element('sdf', version='1.9')
    model = ET.SubElement(sdf, 'model', name='controller_truth_probe')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'link', name='probe')
    library = os.path.join(get_package_prefix('ibvs_controller_test'),
                           'lib', 'libcontroller_truth_system.so')
    plugin = ET.SubElement(model, 'plugin', filename=library,
                           name='ibvs_controller_test::ControllerTruthSystem')
    parameters = {
        'vehicle_model': value('vehicle_model_name'), 'target_model': 'ibvs_target',
        'topic': f'/world/{world}/controller_test/state',
        'moving_target': value('moving_target'), 'target_pattern': value('target_pattern'),
        'target_speed_m_s': value('target_speed_m_s'),
        'radius_x_m': value('figure8_radius_x_m'), 'radius_y_m': value('figure8_radius_y_m'),
        'target_tilt_deg': value('target_tilt_deg'),
    }
    for name, parameter in parameters.items():
        ET.SubElement(plugin, name).text = parameter
    return [Node(
        package='ros_gz_sim', executable='create', name='spawn_controller_truth_probe',
        arguments=['-world', world, '-string', ET.tostring(sdf, encoding='unicode'),
                   '-name', 'controller_truth_probe'], output='screen',
    )]


def generate_launch_description():
    """Select an isolated truth input and record the actual command/feedback path."""
    share = get_package_share_directory('ibvs_control')
    with open(os.path.join(share, 'config', 'vision_direct_interception.yaml'),
              encoding='utf-8') as stream:
        flight = yaml.safe_load(stream)['vision_interception_coordinator']['ros__parameters']
    state_topic = '/interception/controller_test/state'
    truth = Node(
        package='ibvs_control', executable='controller_truth_state',
        name='controller_truth_state', output='screen',
        parameters=[{
            'use_sim_time': True,
            'world_name': LaunchConfiguration('world'),
            'state_topic': state_topic,
            'stop_motion_on_shutdown': True,
            'camera_to_body_rotation': flight['camera_to_body_rotation'],
            'camera_position_b_m': flight['camera_position_b_m'],
        }],
    )
    coordinator = Node(
        package='ibvs_control', executable='controller_truth_coordinator',
        name='vision_interception_coordinator', output='screen',
        parameters=[
            flight,
            os.path.join(share, 'config', 'design_parameters.yaml'),
            {
                'observer_topic': state_topic,
                'enable_flight_commands': ParameterValue(
                    LaunchConfiguration('enable_flight_commands'), value_type=bool,
                ),
                'confirmation_token': LaunchConfiguration('confirmation_token'),
            },
        ],
    )
    bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-o',
            PathJoinSubstitution([
                LaunchConfiguration('results_directory'), 'bags',
                LaunchConfiguration('trial_id'),
            ]),
            '/clock', '/camera/camera_info',
            '/interception/vision/raw_feature', state_topic,
            '/interception/observer/bootstrap', '/interception/observer/reset',
            '/interception/control/debug', '/interception/target/green_confirmed',
            '/world/default/model/ibvs_target/link/target_link/sensor/'
            'target_contact/contact',
            '/fmu/in/offboard_control_mode', '/fmu/in/trajectory_setpoint',
            '/fmu/in/vehicle_rates_setpoint', '/fmu/in/vehicle_command',
            '/fmu/out/vehicle_attitude', '/fmu/out/sensor_combined',
            '/fmu/out/timesync_status', '/fmu/out/vehicle_land_detected',
            '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_odometry',
            '/fmu/out/vehicle_status_v4',
        ],
        output='screen', condition=IfCondition(LaunchConfiguration('record_bag')),
    )
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=coordinator,
        on_exit=[EmitEvent(event=Shutdown(reason='Controller truth test finished'))],
    ))
    return LaunchDescription([
        DeclareLaunchArgument('trial_id', default_value='controller_truth_01'),
        DeclareLaunchArgument('world', default_value='default'),
        DeclareLaunchArgument('vehicle_model_name', default_value='x500_mono_cam_0'),
        DeclareLaunchArgument('moving_target', default_value='false'),
        DeclareLaunchArgument('target_pattern', default_value='circle'),
        DeclareLaunchArgument('target_speed_m_s', default_value='4.0'),
        DeclareLaunchArgument('figure8_radius_x_m', default_value='6.0'),
        DeclareLaunchArgument('figure8_radius_y_m', default_value='6.0'),
        DeclareLaunchArgument('target_tilt_deg', default_value='45.0'),
        DeclareLaunchArgument(
            'results_directory',
            default_value=os.path.expanduser('~/ibvs_ws/results/controller_truth'),
        ),
        DeclareLaunchArgument('enable_flight_commands', default_value='false'),
        DeclareLaunchArgument('confirmation_token', default_value=''),
        DeclareLaunchArgument('record_bag', default_value='true'),
        Node(
            package='ibvs_control', executable='acceleration_recorder',
            name='acceleration_recorder', output='screen',
            parameters=[{
                'use_sim_time': True,
                'trial_id': LaunchConfiguration('trial_id'),
                'world_name': LaunchConfiguration('world'),
                'vehicle_model': LaunchConfiguration('vehicle_model_name'),
                'output_directory': PathJoinSubstitution([
                    LaunchConfiguration('results_directory'), 'acceleration',
                ]),
            }],
        ),
        shutdown, bag, OpaqueFunction(function=spawn_physics_probe), truth, coordinator,
    ])
