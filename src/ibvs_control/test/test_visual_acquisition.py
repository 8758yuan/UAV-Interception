"""Tests for camera-only search without pre-interception alignment."""

import math
from dataclasses import replace

import pytest

from ibvs_control.visual_acquisition import (
    VisualAcquisition,
    VisualAcquisitionConfig,
    VisualAcquisitionPhase,
)


def _config() -> VisualAcquisitionConfig:
    return VisualAcquisitionConfig(
        search_yaw_rate_rad_s=0.2,
        search_vertical_amplitude_m=0.3,
        search_vertical_period_s=4.0,
        desired_image_xy=(0.0, 0.0),
        target_loss_timeout_s=0.3,
    )


def test_search_scans_relative_to_current_yaw_without_target_state() -> None:
    acquisition = VisualAcquisition(_config())
    first = acquisition.step(
        now_s=1.0,
        current_yaw_rad=0.7,
        target_detected=False,
    )
    command = acquisition.step(
        now_s=2.0,
        current_yaw_rad=0.7,
        target_detected=False,
    )
    assert first.phase == VisualAcquisitionPhase.SEARCH
    assert not command.ready
    assert command.yaw_setpoint_rad == pytest.approx(0.9)


def test_target_detection_stops_scan_at_current_vehicle_yaw() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=0.4,
        target_detected=False,
    )
    searching = acquisition.step(
        now_s=2.0,
        current_yaw_rad=0.7,
        target_detected=False,
    )
    detected = acquisition.step(
        now_s=2.1,
        current_yaw_rad=0.7,
        target_detected=True,
        x_norm=0.2,
    )

    assert searching.yaw_setpoint_rad == pytest.approx(0.8)
    assert detected.phase == VisualAcquisitionPhase.READY
    assert detected.ready
    assert detected.yaw_setpoint_rad == pytest.approx(0.7)
    assert detected.vertical_offset_m == pytest.approx(
        searching.vertical_offset_m
    )


def test_search_scans_altitude_without_target_state() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=1.0,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    rising = acquisition.step(
        now_s=2.0,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    falling = acquisition.step(
        now_s=4.0,
        current_yaw_rad=0.0,
        target_detected=False,
    )

    assert rising.vertical_offset_m == pytest.approx(0.3)
    assert falling.vertical_offset_m == pytest.approx(-0.3)


@pytest.mark.parametrize('image_xy', [
    (0.0, 0.0), (0.35, -0.28), (-0.6, -0.1), (0.9, 0.5),
])
def test_first_detection_is_ready_without_alignment_or_dwell(image_xy) -> None:
    acquisition = VisualAcquisition(_config())
    for t in (0.0, 0.1):
        command = acquisition.step(
            now_s=t,
            current_yaw_rad=0.7,
            target_detected=True,
            x_norm=image_xy[0],
            y_norm=image_xy[1],
        )
        assert command.phase == VisualAcquisitionPhase.READY
        assert command.ready
        assert command.yaw_setpoint_rad == pytest.approx(0.7)
        assert command.vertical_offset_m == pytest.approx(0.0)


def test_target_motion_does_not_restart_alignment_or_change_hold() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=False,
    )
    acquisition.step(
        now_s=1.0, current_yaw_rad=0.2, target_detected=False,
    )
    detected = acquisition.step(
        now_s=1.1, current_yaw_rad=0.25, target_detected=True,
    )
    moving = acquisition.step(
        now_s=1.2, current_yaw_rad=0.3, target_detected=True,
        x_norm=0.9, y_norm=-0.7,
    )
    assert moving.phase == VisualAcquisitionPhase.READY
    assert moving.ready
    assert moving.yaw_setpoint_rad == pytest.approx(detected.yaw_setpoint_rad)
    assert moving.vertical_offset_m == pytest.approx(0.3)


def test_designed_image_position_only_changes_error_diagnostic() -> None:
    config = replace(_config(), desired_image_xy=(0.0, -0.7))
    command = VisualAcquisition(config).step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=True,
    )
    assert command.ready
    assert command.image_error == pytest.approx(0.7)
    assert command.yaw_setpoint_rad == pytest.approx(0.0)
    assert command.vertical_offset_m == pytest.approx(0.0)


def test_detection_recovers_readiness_without_waiting_after_dropout() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=True,
    )
    dropout = acquisition.step(
        now_s=0.1, current_yaw_rad=0.0, target_detected=False,
    )
    assert not dropout.ready
    recovered = acquisition.step(
        now_s=0.15, current_yaw_rad=0.0, target_detected=True,
    )
    assert recovered.phase == VisualAcquisitionPhase.READY
    assert recovered.ready


def test_target_loss_restarts_search_after_existing_timeout() -> None:
    acquisition = VisualAcquisition(_config())
    ready = acquisition.step(
        now_s=0.0,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    assert ready.ready

    dropout = acquisition.step(
        now_s=0.1,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    assert dropout.phase == VisualAcquisitionPhase.READY
    assert not dropout.ready

    searching = acquisition.step(
        now_s=0.31,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    assert searching.phase == VisualAcquisitionPhase.SEARCH
    assert not searching.ready


def test_search_wraps_yaw_setpoint() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=math.pi - 0.01,
        target_detected=False,
    )
    command = acquisition.step(
        now_s=1.0,
        current_yaw_rad=math.pi - 0.01,
        target_detected=False,
    )
    assert -math.pi <= command.yaw_setpoint_rad <= math.pi
