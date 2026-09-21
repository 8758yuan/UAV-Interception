"""Tests for camera-only search and pre-interception alignment."""

import math

import pytest

from ibvs_control.visual_acquisition import (
    VisualAcquisition,
    VisualAcquisitionConfig,
    VisualAcquisitionPhase,
)


def _config() -> VisualAcquisitionConfig:
    return VisualAcquisitionConfig(
        search_yaw_rate_rad_s=0.2,
        align_yaw_gain_rad_s=1.2,
        align_vertical_gain_m_s=0.4,
        center_error=0.08,
        release_error=0.14,
        settle_time_s=0.5,
        target_loss_timeout_s=0.3,
        maximum_vertical_offset_m=0.6,
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
    assert command.yaw_setpoint_rad == pytest.approx(0.9)


def test_alignment_uses_pixel_sign_and_requires_settle_time() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=1.0,
        target_detected=True,
        x_norm=-0.2,
        y_norm=-0.1,
    )
    correcting = acquisition.step(
        now_s=0.5,
        current_yaw_rad=1.0,
        target_detected=True,
        x_norm=-0.2,
        y_norm=-0.1,
    )
    assert correcting.phase == VisualAcquisitionPhase.ALIGN
    assert correcting.yaw_setpoint_rad < 1.0
    assert correcting.vertical_offset_m < 0.0
    acquisition.step(
        now_s=0.6,
        current_yaw_rad=correcting.yaw_setpoint_rad,
        target_detected=True,
        x_norm=0.02,
        y_norm=-0.01,
    )
    ready = acquisition.step(
        now_s=1.11,
        current_yaw_rad=correcting.yaw_setpoint_rad,
        target_detected=True,
        x_norm=0.02,
        y_norm=-0.01,
    )
    assert ready.ready
    assert ready.phase == VisualAcquisitionPhase.READY


def test_lost_alignment_returns_to_search() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=0.0,
        target_detected=True,
        x_norm=0.2,
    )
    command = acquisition.step(
        now_s=0.31,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    assert command.phase == VisualAcquisitionPhase.SEARCH
    assert not command.ready


def test_ready_lock_is_revoked_immediately_when_target_disappears() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    acquisition.step(
        now_s=0.1,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    ready = acquisition.step(
        now_s=0.61,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    assert ready.ready

    dropout = acquisition.step(
        now_s=0.70,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    assert dropout.phase == VisualAcquisitionPhase.READY
    assert not dropout.ready

    searching = acquisition.step(
        now_s=0.92,
        current_yaw_rad=0.0,
        target_detected=False,
    )
    assert searching.phase == VisualAcquisitionPhase.SEARCH
    assert not searching.ready


def test_ready_lock_returns_to_alignment_when_target_moves() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    acquisition.step(
        now_s=0.1,
        current_yaw_rad=0.0,
        target_detected=True,
    )
    acquisition.step(
        now_s=0.61,
        current_yaw_rad=0.0,
        target_detected=True,
    )

    command = acquisition.step(
        now_s=0.62,
        current_yaw_rad=0.0,
        target_detected=True,
        x_norm=0.15,
    )
    assert command.phase == VisualAcquisitionPhase.ALIGN
    assert not command.ready


def test_configuration_rejects_inverted_hysteresis() -> None:
    config = _config()
    with pytest.raises(ValueError, match='release_error'):
        VisualAcquisitionConfig(
            **{
                **config.__dict__,
                'release_error': config.center_error,
            }
        ).validate()


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
