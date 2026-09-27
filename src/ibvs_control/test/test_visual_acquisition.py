"""Tests for camera-only search and barrier-aware acquisition."""

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
        align_yaw_gain_rad_s=1.2,
        align_vertical_gain_m_s=0.4,
        k_b=1.0 - math.cos(math.radians(30.0)),
        desired_image_xy=(0.0, 0.0),
        start_barrier_margin=0.04,
        release_barrier_margin=0.025,
        settle_time_s=0.5,
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
    assert command.yaw_setpoint_rad == pytest.approx(0.9)


def test_target_detection_stops_scan_at_current_vehicle_yaw() -> None:
    """Acquisition switches to alignment without continuing a stale scan."""
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
    assert detected.phase == VisualAcquisitionPhase.ACQUIRE
    assert detected.yaw_setpoint_rad == pytest.approx(0.7)


def test_search_scans_altitude_without_target_state() -> None:
    """Search changes height as well as yaw when the target is not visible."""
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


def test_alignment_uses_pixel_sign_and_requires_settle_time() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0,
        current_yaw_rad=1.0,
        target_detected=True,
        x_norm=-0.6,
        y_norm=-0.1,
    )
    correcting = acquisition.step(
        now_s=0.5,
        current_yaw_rad=1.0,
        target_detected=True,
        x_norm=-0.6,
        y_norm=-0.1,
    )
    assert correcting.phase == VisualAcquisitionPhase.ACQUIRE
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


def test_off_center_target_starts_without_precise_alignment() -> None:
    """A visible target inside the barrier can launch without yaw or climb."""
    acquisition = VisualAcquisition(_config())
    detected, settling, ready = (
        acquisition.step(
            now_s=t,
            current_yaw_rad=0.7,
            target_detected=True,
            x_norm=0.35,
            y_norm=-0.28,
        )
        for t in (0.0, 0.1, 0.61)
    )
    assert detected.phase == VisualAcquisitionPhase.ACQUIRE
    assert not settling.ready
    assert ready.phase == VisualAcquisitionPhase.READY
    assert ready.ready
    assert ready.image_error > 0.04
    assert ready.yaw_setpoint_rad == pytest.approx(0.7)
    assert ready.vertical_offset_m == pytest.approx(0.0)


def test_dwell_restarts_if_target_leaves_start_margin() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=True
    )
    acquisition.step(
        now_s=0.1, current_yaw_rad=0.0, target_detected=True
    )
    crossing = acquisition.step(
        now_s=0.3, current_yaw_rad=0.0, target_detected=True,
        x_norm=0.5,
    )
    assert not crossing.ready
    acquisition.step(
        now_s=0.4, current_yaw_rad=0.0, target_detected=True
    )
    early = acquisition.step(
        now_s=0.7, current_yaw_rad=0.0, target_detected=True
    )
    ready = acquisition.step(
        now_s=0.91, current_yaw_rad=0.0, target_detected=True
    )
    assert not early.ready
    assert ready.ready


def test_ready_hysteresis_allows_target_motion_within_safe_margin() -> None:
    acquisition = VisualAcquisition(_config())
    for t in (0.0, 0.1, 0.61):
        acquisition.step(
            now_s=t, current_yaw_rad=0.0, target_detected=True
        )
    moving = acquisition.step(
        now_s=0.62, current_yaw_rad=0.0, target_detected=True,
        x_norm=0.5,
    )
    assert moving.phase == VisualAcquisitionPhase.READY
    assert moving.ready


def test_barrier_uses_designed_los_not_image_origin() -> None:
    config = replace(_config(), desired_image_xy=(0.0, -0.5))
    acquisition = VisualAcquisition(config)
    acquisition.step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=True
    )
    correcting = acquisition.step(
        now_s=0.1, current_yaw_rad=0.0, target_detected=True
    )
    assert correcting.phase == VisualAcquisitionPhase.ACQUIRE
    assert correcting.vertical_offset_m > 0.0
    assert not correcting.ready


def test_missing_frames_do_not_count_toward_stable_dwell() -> None:
    acquisition = VisualAcquisition(_config())
    acquisition.step(
        now_s=0.0, current_yaw_rad=0.0, target_detected=True
    )
    acquisition.step(
        now_s=0.1, current_yaw_rad=0.0, target_detected=True
    )
    acquisition.step(
        now_s=0.2, current_yaw_rad=0.0, target_detected=False
    )
    acquisition.step(
        now_s=0.25, current_yaw_rad=0.0, target_detected=True
    )
    early = acquisition.step(
        now_s=0.61, current_yaw_rad=0.0, target_detected=True
    )
    ready = acquisition.step(
        now_s=0.71, current_yaw_rad=0.0, target_detected=True
    )
    assert not early.ready
    assert ready.ready


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
        x_norm=0.55,
    )
    assert command.phase == VisualAcquisitionPhase.ACQUIRE
    assert not command.ready


def test_configuration_rejects_inverted_hysteresis() -> None:
    config = _config()
    with pytest.raises(ValueError, match='barrier margins'):
        VisualAcquisitionConfig(
            **{
                **config.__dict__,
                'release_barrier_margin': config.start_barrier_margin,
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
