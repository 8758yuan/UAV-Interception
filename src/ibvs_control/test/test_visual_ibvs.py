"""Tests for the target-state-free visual servo law."""

import inspect
import math

import numpy as np
import pytest

from ibvs_control.visual_ibvs import (
    VisualIbvsConfig,
    compute_visual_ibvs,
    recent_close_target_lost,
    terminal_visual_ready,
)


def _config() -> VisualIbvsConfig:
    return VisualIbvsConfig(
        mass_kg=2.0,
        thrust_max_n=27.0,
        k_b=0.5,
        los_rate_gain=2.5,
        image_center_rate_gain=1.0,
        attitude_rate_gain=0.8,
        speed_gain=1.2,
        cruise_speed_m_s=2.0,
        terminal_speed_m_s=2.8,
        max_approach_acceleration_m_s2=1.8,
        max_approach_deceleration_m_s2=1.0,
        transverse_velocity_gain=1.5,
        max_transverse_acceleration_m_s2=2.0,
        vertical_velocity_gain=2.0,
        max_vertical_correction_m_s2=2.0,
        terminal_area_ratio=0.16,
        full_speed_image_error=0.08,
        stop_approach_image_error=0.30,
        max_command_tilt_rad=math.radians(16.0),
    )


def test_controller_api_has_no_target_state_input() -> None:
    names = set(inspect.signature(compute_visual_ibvs).parameters)
    forbidden = {
        'target_position',
        'target_velocity',
        'relative_position',
        'relative_velocity',
        'distance',
        'time_to_collision',
    }
    assert names.isdisjoint(forbidden)
    assert {'x_norm', 'y_norm', 'area_ratio'} <= names


def test_centered_target_commands_forward_approach() -> None:
    result = compute_visual_ibvs(
        0.0,
        0.0,
        0.001,
        np.eye(3),
        (0.0, 0.0, 0.0),
        _config(),
    )
    assert result.los_b == pytest.approx((1.0, 0.0, 0.0))
    assert result.omega_los_b == pytest.approx((0.0, 0.0, 0.0))
    assert result.acceleration_d_e[0] > 0.0
    assert result.thrust_n > 0.0


def test_image_error_directly_sets_los_correction_sign() -> None:
    result = compute_visual_ibvs(
        0.2,
        0.1,
        0.002,
        np.eye(3),
        (0.0, 0.0, 0.0),
        _config(),
    )
    # Target right/below -> body-FLU yaw right (negative z) and pitch down
    # (positive y), matching paper equation (13).
    assert result.omega_los_b[1] > 0.0
    assert result.omega_los_b[2] < 0.0


def test_image_area_schedules_terminal_speed_without_range() -> None:
    far = compute_visual_ibvs(
        0.0, 0.0, 0.001, np.eye(3), (0.0, 0.0, 0.0), _config()
    )
    close = compute_visual_ibvs(
        0.0, 0.0, 0.16, np.eye(3), (0.0, 0.0, 0.0), _config()
    )
    assert close.desired_approach_speed_m_s > far.desired_approach_speed_m_s
    assert close.desired_approach_speed_m_s == pytest.approx(2.8)


def test_image_error_slows_approach_before_target_reaches_edge() -> None:
    centered = compute_visual_ibvs(
        0.0, 0.0, 0.01, np.eye(3), (0.0, 0.0, 0.0), _config()
    )
    offset = compute_visual_ibvs(
        0.2, 0.0, 0.01, np.eye(3), (0.0, 0.0, 0.0), _config()
    )
    edge = compute_visual_ibvs(
        0.31, 0.0, 0.01, np.eye(3), (0.0, 0.0, 0.0), _config()
    )
    assert centered.alignment_factor == pytest.approx(1.0)
    assert 0.0 < offset.alignment_factor < 1.0
    assert edge.alignment_factor == pytest.approx(0.0)
    assert centered.desired_approach_speed_m_s > (
        offset.desired_approach_speed_m_s
    )
    assert edge.desired_approach_speed_m_s == pytest.approx(0.0)


def test_own_vertical_velocity_damps_altitude_loss() -> None:
    result = compute_visual_ibvs(
        0.0, 0.0, 0.01, np.eye(3), (0.0, 0.0, -0.5), _config()
    )
    assert result.vertical_correction_m_s2 == pytest.approx(1.0)
    assert result.transverse_correction_e[2] == pytest.approx(0.75)
    assert result.acceleration_d_e[2] == pytest.approx(1.75)


def test_own_cross_track_velocity_is_damped_from_camera_los() -> None:
    result = compute_visual_ibvs(
        0.0, 0.0, 0.01, np.eye(3), (0.0, 0.6, 0.0), _config()
    )
    assert result.approach_speed_m_s == pytest.approx(0.0)
    assert result.transverse_velocity_e == pytest.approx((0.0, 0.6, 0.0))
    assert result.transverse_correction_e == pytest.approx((0.0, -0.9, 0.0))
    assert result.acceleration_d_e[1] < 0.0


def test_terminal_decision_requires_scale_and_centering() -> None:
    assert terminal_visual_ready(0.16, 0.1, -0.1, 0.16, 0.35)
    assert not terminal_visual_ready(0.159, 0.0, 0.0, 0.16, 0.35)
    assert not terminal_visual_ready(0.20, 0.4, 0.0, 0.16, 0.35)


def test_recent_close_loss_is_a_visual_only_terminal_cue() -> None:
    assert recent_close_target_lost(0.08, 0.2, 0.1, 0.05, 0.04, 0.6, 0.3)
    assert not recent_close_target_lost(
        0.01, 0.2, 0.1, 0.05, 0.04, 0.6, 0.3
    )
    assert not recent_close_target_lost(
        0.08, 0.2, 0.1, 0.31, 0.04, 0.6, 0.3
    )
