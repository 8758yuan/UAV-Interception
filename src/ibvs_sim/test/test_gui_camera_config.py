"""Regression checks for the manually controlled Gazebo user camera."""

from pathlib import Path


def test_third_person_camera_remains_under_interactive_mouse_control() -> None:
    config = (
        Path(__file__).parents[1]
        / 'config'
        / 'camera_3d_and_first_person.config'
    ).read_text(encoding='utf-8')

    assert '<view_controller>orbit</view_controller>' in config
    assert 'filename="InteractiveViewControl"' in config
    assert 'filename="CameraTracking"' not in config


def test_px4_start_instructions_disable_automatic_camera_follow() -> None:
    instructions = (
        Path(__file__).parents[3] / '启动.md'
    ).read_text(encoding='utf-8')

    assert 'export PX4_GZ_NO_FOLLOW=1' in instructions
