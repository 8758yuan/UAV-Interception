"""Small regression tests for coordinator message-derived quantities."""

from types import SimpleNamespace

import pytest

from ibvs_control.truth_interception_coordinator import (
    _has_expected_target_contact,
    _recovery_target_z_ned,
    _relative_distance,
    _terminal_attack_acceleration,
)


def test_contact_requires_target_and_expected_vehicle() -> None:
    expected = SimpleNamespace(
        collision1=SimpleNamespace(name='target_collision'),
        collision2=SimpleNamespace(name='vehicle::rotor_collision'),
    )
    ground = SimpleNamespace(
        collision1=SimpleNamespace(name='target_collision'),
        collision2=SimpleNamespace(name='ground_plane::collision'),
    )
    assert _has_expected_target_contact(
        SimpleNamespace(contacts=[expected]),
        'target_collision',
        'vehicle::',
    )
    assert not _has_expected_target_contact(
        SimpleNamespace(contacts=[ground]),
        'target_collision',
        'vehicle::',
    )


def test_relative_distance_is_derived_from_p_r() -> None:
    """Relative state intentionally has no redundant distance field."""
    message = SimpleNamespace(
        p_r=SimpleNamespace(x=3.0, y=4.0, z=12.0)
    )
    assert _relative_distance(message) == pytest.approx(13.0)


def test_invalid_relative_position_returns_safety_sentinel() -> None:
    """Nonfinite truth data must fail the hit test safely."""
    message = SimpleNamespace(
        p_r=SimpleNamespace(x=float('nan'), y=0.0, z=0.0)
    )
    assert _relative_distance(message) == 1e9


def test_terminal_attack_accepts_ros_tuple_vectors() -> None:
    acceleration = _terminal_attack_acceleration(
        (1.0, 0.0, 0.0),
        distance_m=2.0,
        activation_distance_m=2.5,
        acceleration_m_s2=2.5,
    )
    assert acceleration == pytest.approx((2.5, 0.0, 0.0))


def test_terminal_attack_is_zero_outside_activation_distance() -> None:
    acceleration = _terminal_attack_acceleration(
        (1.0, 0.0, 0.0),
        distance_m=3.0,
        activation_distance_m=2.5,
        acceleration_m_s2=2.5,
    )
    assert acceleration == pytest.approx((0.0, 0.0, 0.0))


def test_recovery_ned_height_climbs_when_below_safe_height() -> None:
    """In NED, current z=-1 below target z=-2 requires a smaller command."""
    assert _recovery_target_z_ned(-1.0, -2.0) == pytest.approx(-2.0)


def test_recovery_ned_height_never_descends_when_already_high() -> None:
    """In NED, retaining z=-3 instead of target z=-2 prevents descent."""
    assert _recovery_target_z_ned(-3.0, -2.0) == pytest.approx(-3.0)
