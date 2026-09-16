"""Architecture regressions for the command-producing visual path."""

import inspect
from pathlib import Path

from ibvs_control import vision_interception_coordinator


def test_visual_coordinator_has_no_target_state_or_contact_subscription() -> None:
    source = inspect.getsource(vision_interception_coordinator)
    forbidden = (
        'RelativeState',
        'ros_gz_interfaces',
        'target_odometry',
        'target_contact',
        'relative_state_topic',
        'truth_state',
    )
    for token in forbidden:
        assert token not in source


def test_vision_launch_selects_visual_coordinator() -> None:
    launch_path = (
        Path(__file__).parents[1]
        / 'launch'
        / 'vision_direct_interception.launch.py'
    )
    source = launch_path.read_text(encoding='utf-8')
    assert "executable='vision_interception_coordinator'" in source
    assert "executable='truth_interception_coordinator'" not in source


def test_truth_coordinator_is_not_an_installed_executable() -> None:
    setup_path = Path(__file__).parents[1] / 'setup.py'
    source = setup_path.read_text(encoding='utf-8')
    assert "'truth_interception_coordinator = '" not in source
