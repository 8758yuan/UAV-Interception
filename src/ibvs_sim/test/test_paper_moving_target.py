"""Tests for the paper moving target confined to the Gazebo plant."""

import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
import pytest

from ibvs_sim.paper_moving_target import PaperTargetTrajectory


@pytest.mark.parametrize('pattern', ('figure8', 'circle'))
def test_trajectory_starts_at_spawn_pose_and_holds_requested_speed(pattern) -> None:
    trajectory = PaperTargetTrajectory(
        pattern=pattern,
        origin=(12.0, 0.0, 3.0),
        speed_m_s=5.0,
        radius_x_m=4.0,
        radius_y_m=2.0 if pattern == 'figure8' else 4.0,
    )
    start = trajectory.sample(0.0)
    assert start.position == pytest.approx((12.0, 0.0, 3.0))
    for time_s in (0.1, 0.7, 1.9, 4.2):
        sample = trajectory.sample(time_s)
        assert math.hypot(*sample.velocity[:2]) == pytest.approx(5.0)
        assert sample.velocity[2] == 0.0


def test_figure8_is_periodic_and_crosses_its_origin() -> None:
    trajectory = PaperTargetTrajectory(
        pattern='figure8',
        origin=(12.0, 0.0, 3.0),
        speed_m_s=7.5,
        radius_x_m=4.0,
        radius_y_m=2.0,
    )
    period_s = trajectory.path_length_m / trajectory.speed_m_s
    assert trajectory.sample(period_s).position == pytest.approx(
        trajectory.sample(0.0).position,
        abs=1e-8,
    )
    assert trajectory.sample(0.5 * period_s).position == pytest.approx(
        (12.0, 0.0, 3.0),
        abs=1e-6,
    )


@pytest.mark.parametrize('pattern', ('static', 'circle', 'figure8', 'line'))
def test_scenario_spawns_exactly_one_motion_driver_only_for_moving_targets(pattern):
    launch_root = Path(__file__).parents[1] / 'launch'
    path = launch_root / ('p4_vision_monitor.launch.py' if pattern == 'static'
                          else 'p7_moving_target.launch.py')
    spec = spec_from_file_location('target_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations['target_pattern'] = pattern
    description = module.generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    if pattern == 'line':
        context.launch_configurations.update({
            'target_x': '30.0', 'target_y': '0.0', 'target_z': '12.0',
            'target_direction_x': '-2.0', 'target_direction_y': '-3.0',
            'target_direction_z': '1.0',
        })
    if pattern != 'static':
        include, = [action for action in description.entities
                    if isinstance(action, IncludeLaunchDescription)]
        for name, value in include.launch_arguments:
            context.launch_configurations[name] = perform_substitutions(
                context, normalize_to_list_of_substitutions(value))
        spec = spec_from_file_location('target_base_launch', launch_root / 'p4_vision_monitor.launch.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        for action in module.generate_launch_description().entities:
            if isinstance(action, DeclareLaunchArgument):
                action.execute(context)
    target, = module.spawn_target(context)
    command = (perform_substitutions(context, normalize_to_list_of_substitutions(part))
               for part in target.cmd)
    next(argument for argument in command if argument == '-string')
    model = ET.fromstring(next(command)).find('model')
    plugins = model.findall('plugin')
    if pattern == 'static':
        assert plugins == []
    else:
        plugin, = plugins
        assert plugin.get('name') == 'ibvs_sim_physics::PaperTargetSystem'
        assert Path(plugin.get('filename')).is_file()
        assert plugin.findtext('target_pattern') == pattern
        assert float(plugin.findtext('target_speed_m_s')) == 5.0
        assert plugin.findtext('start_on_interception') == (
            'true' if pattern == 'line' else 'false')
        if pattern == 'line':
            assert tuple(float(plugin.findtext('target_direction_' + axis))
                         for axis in 'xyz') == (-2.0, -3.0, 1.0)


def test_invalid_moving_target_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        PaperTargetTrajectory(
            pattern='random',
            origin=(12.0, 0.0, 3.0),
            speed_m_s=5.0,
            radius_x_m=4.0,
            radius_y_m=2.0,
        )


def test_tilted_circle_radius_speed_and_direction() -> None:
    trajectory = PaperTargetTrajectory(
        pattern='circle', origin=(12.0, 0.0, 4.0), speed_m_s=5.0,
        radius_x_m=6.0, radius_y_m=6.0, tilt_deg=45.0,
    )
    offset = 6.0 / math.sqrt(2.0)
    centre = (12.0 - offset, 0.0, 4.0 - offset)
    period = trajectory.path_length_m / trajectory.speed_m_s
    for fraction in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        sample = trajectory.sample(fraction * period)
        x, _, z = sample.position
        assert math.dist(sample.position, centre) == pytest.approx(6.0)
        assert z - 4.0 == pytest.approx(x - 12.0)
        assert math.hypot(*sample.velocity) == pytest.approx(5.0)
    assert trajectory.sample(0.0).position == pytest.approx((12.0, 0.0, 4.0))
    assert trajectory.sample(period).position == pytest.approx((12.0, 0.0, 4.0))
    assert trajectory.sample(0.5 * period).position[2] == pytest.approx(
        4.0 - 2.0 * offset,
    )


@pytest.mark.parametrize('direction, velocity', (
    ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    ((-1.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
    ((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
    ((0.0, -1.0, 0.0), (0.0, -2.0, 0.0)),
    ((-3.0, 4.0, 0.0), (-1.2, 1.6, 0.0)),
    ((-2.0, -3.0, 6.0), (-4.0 / 7.0, -6.0 / 7.0, 12.0 / 7.0)),
    ((0.0, 0.0, 5.0), (0.0, 0.0, 2.0)),
))
def test_line_has_constant_world_velocity_without_wrapping(direction, velocity):
    origin = (30.0, 0.0, 12.0)
    trajectory = PaperTargetTrajectory(
        pattern='line', origin=origin, speed_m_s=2.0, direction=direction,
        radius_x_m=0.0, radius_y_m=0.0, tilt_deg=45.0,
    )
    for time_s in (0.0, 0.1, 2.0, 100.0):
        sample = trajectory.sample(time_s)
        assert sample.position == pytest.approx(tuple(
            start + component * time_s for start, component in zip(origin, velocity)))
        assert sample.velocity == pytest.approx(velocity)
        assert math.hypot(*sample.velocity) == pytest.approx(2.0)


@pytest.mark.parametrize('direction', (
    (0.0, 0.0, 0.0), (math.nan, 1.0, 0.0), (math.inf, 0.0, 1.0),
    (1.0, 0.0), (1.0, 0.0, 0.0, 0.0),
))
def test_line_rejects_invalid_direction(direction):
    with pytest.raises(ValueError, match='direction'):
        PaperTargetTrajectory('line', (30.0, 0.0, 12.0), 2.0, direction=direction)
