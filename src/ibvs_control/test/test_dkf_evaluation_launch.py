"""Verify that normal trial launches include evaluation without an opt-in."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest


def load_launch(filename):
    path = Path(__file__).parents[1] / 'launch' / filename
    spec = spec_from_file_location('evaluation_launch', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('filename', [
    'vision_direct_interception_trial.launch.py',
    'vision_direct_interception.launch.py',
])
def test_normal_launch_starts_evaluation(filename, tmp_path):
    module = load_launch(filename)
    description = module.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations['trial_id'] = 'automatic_test'
    context.launch_configurations['results_directory'] = str(tmp_path)
    evaluation, = [action for action in description.entities
                   if isinstance(action, Node) and action.node_executable == 'dkf_evaluation']
    assert evaluation.condition is None
    parameters, = evaluate_parameters(context, evaluation._Node__parameters)
    assert parameters['use_sim_time'] is True
    assert parameters['vehicle_model'] == 'x500_mono_cam_0'
    assert parameters['camera_position_b_m'] == (0.14, 0.0, 0.0)
    if filename.startswith('vision_direct_interception_trial'):
        assert parameters['output_directory'] == str(tmp_path / 'dkf')
        assert parameters['run_id'] == 'automatic_test'
        # Evaluation is independent of optional rosbag recording, and shares
        # the launch shutdown triggered by the existing coordinator exit.
        assert any(isinstance(action, RegisterEventHandler)
                   for action in description.entities)
        context.launch_configurations['record_bag'] = 'false'
        context.launch_configurations['minimum_bag_free_space_gb'] = '1.0'
        module._prepare_directories(context)
        (tmp_path / 'dkf' / 'automatic_test').mkdir(parents=True)
        with pytest.raises(FileExistsError, match='DKF results already exist'):
            module._prepare_directories(context)


def test_trial_exit_saves_report_without_manual_recorder(monkeypatch, tmp_path):
    """Run real recorder/launch shutdown with dummy flight processes only."""
    import csv
    import sys

    from launch import LaunchDescription, LaunchService
    from launch.actions import ExecuteProcess, SetLaunchConfiguration

    module = load_launch('vision_direct_interception_trial.launch.py')
    real_node = module.Node

    def simulation_node(**kwargs):
        if kwargs['executable'] == 'dkf_evaluation':
            return real_node(**kwargs)
        duration = 2 if kwargs['executable'] == 'vision_interception_coordinator' else 10
        return ExecuteProcess(cmd=[sys.executable, '-c',
                                   f'import time; time.sleep({duration})'])

    monkeypatch.setattr(module, 'Node', simulation_node)
    description = module.generate_launch_description()
    service = LaunchService()
    service.include_launch_description(LaunchDescription([
        SetLaunchConfiguration('results_directory', str(tmp_path)),
        SetLaunchConfiguration('trial_id', 'automatic_exit'),
        SetLaunchConfiguration('record_bag', 'false'),
        *description.entities,
    ]))
    assert service.run() == 0
    directory = tmp_path / 'dkf' / 'automatic_exit'
    assert (directory / 'states.csv').exists()
    assert (directory / '测试结果.html').exists()
    assert (directory / '每0.5秒对比.xlsx').exists()
    # Even without DKF input, normal launch shutdown must finalize all states.
    rows = list(csv.DictReader((directory / 'metrics.csv').open()))
    assert len(rows) == 18
    assert all(row['count'] == '0' for row in rows)
