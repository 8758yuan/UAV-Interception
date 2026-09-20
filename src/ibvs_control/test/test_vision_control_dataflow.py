"""Architecture regressions for the command-producing visual path."""

import inspect
from pathlib import Path

from ibvs_control import vision_interception_coordinator
from ibvs_control import paper_state_observer_node


def test_visual_coordinator_has_no_target_state_subscription() -> None:
    source = inspect.getsource(vision_interception_coordinator)
    forbidden = (
        'RelativeState',
        'ros_gz_interfaces',
        'target_odometry',
        'relative_state_topic',
        'truth_state',
    )
    for token in forbidden:
        assert token not in source


def test_paper_controller_uses_observer_as_command_authority() -> None:
    source = inspect.getsource(vision_interception_coordinator)
    assert 'ObserverState' in source
    assert 'compute_outer_loop' in source
    assert 'compute_inner_loop' in source
    assert 'compute_visual_ibvs' not in source
    assert 'message.body_rate = True' in source
    assert 'target_contact_topic' in source


def test_vision_launch_selects_visual_coordinator() -> None:
    launch_path = (
        Path(__file__).parents[1]
        / 'launch'
        / 'vision_direct_interception.launch.py'
    )
    source = launch_path.read_text(encoding='utf-8')
    assert "executable='vision_interception_coordinator'" in source
    assert "executable='paper_state_observer'" in source
    assert "executable='truth_interception_coordinator'" not in source


def test_truth_coordinator_is_not_an_installed_executable() -> None:
    setup_path = Path(__file__).parents[1] / 'setup.py'
    source = setup_path.read_text(encoding='utf-8')
    assert "'truth_interception_coordinator = '" not in source


def test_legacy_target_state_pipeline_is_absent() -> None:
    """Removed target-state nodes, messages, and launches cannot return."""
    workspace_src = Path(__file__).parents[2]
    removed_paths = (
        'ibvs_control/ibvs_control/truth_state_node.py',
        'ibvs_control/ibvs_control/truth_interception_coordinator.py',
        'ibvs_control/ibvs_control/controller_shadow.py',
        'ibvs_control/launch/controller_shadow.launch.py',
        'ibvs_sim/launch/p2_truth_monitor.launch.py',
        'interception_interfaces/msg/RelativeState.msg',
    )
    for relative_path in removed_paths:
        assert not (workspace_src / relative_path).exists()


def test_simulation_target_does_not_publish_odometry() -> None:
    """The simulated target exposes imagery/contact, not world-state data."""
    sim_root = Path(__file__).parents[2] / 'ibvs_sim'
    model = (sim_root / 'models' / 'static_target.sdf').read_text(
        encoding='utf-8'
    )
    launch = (sim_root / 'launch' / 'p4_vision_monitor.launch.py').read_text(
        encoding='utf-8'
    )
    assert 'OdometryPublisher' not in model
    assert 'odom_topic' not in model
    assert 'target_odometry' not in launch
    assert 'nav_msgs/msg/Odometry' not in launch


def test_moving_target_is_confined_to_gazebo_set_pose() -> None:
    """The paper trajectory may move the plant but cannot feed the algorithm."""
    sim_root = Path(__file__).parents[2] / 'ibvs_sim'
    mover = (sim_root / 'ibvs_sim' / 'paper_moving_target.py').read_text(
        encoding='utf-8'
    )
    launch = (sim_root / 'launch' / 'p7_moving_target.launch.py').read_text(
        encoding='utf-8'
    )
    assert "set_pose_service = f'/world/{world_name}/set_pose'" in mover
    assert 'Odometry' not in mover
    assert 'Publisher' not in mover
    assert "default_value='figure8'" in launch
    assert "default_value='5.0'" in launch


def test_trial_bag_has_no_target_state_topic() -> None:
    """Trial recording cannot silently preserve the removed target state."""
    trial_launch = (
        Path(__file__).parents[1]
        / 'launch'
        / 'vision_direct_interception_trial.launch.py'
    ).read_text(encoding='utf-8')
    assert '/interception/truth/relative_state' not in trial_launch
    assert '/model/ibvs_target/odometry' not in trial_launch


def test_observer_uses_only_onboard_imu_attitude_and_image() -> None:
    """The estimator must not recreate the deleted target-state pipeline."""
    source = inspect.getsource(paper_state_observer_node)
    required = ('SensorCombined', 'VehicleAttitude', 'VisionFeature')
    forbidden = ('Odometry', 'RelativeState', 'target_position', 'target_velocity')
    for token in required:
        assert token in source
    for token in forbidden:
        assert token not in source


def test_trial_launches_and_records_observer() -> None:
    """Every closed-loop trial captures observer output for validation."""
    trial_launch = (
        Path(__file__).parents[1]
        / 'launch'
        / 'vision_direct_interception_trial.launch.py'
    ).read_text(encoding='utf-8')
    assert "executable='paper_state_observer'" in trial_launch
    assert "'/interception/observer/state'" in trial_launch
    assert "'/interception/observer/reset'" in trial_launch
    assert "'/fmu/out/sensor_combined'" in trial_launch
    assert "DeclareLaunchArgument('speed_limit_m_s'" in trial_launch
    assert "'max_horizontal_distance_m': ParameterValue" in trial_launch
