"""Tests for the reusable interception safety gate."""

import inspect

from ibvs_control.interception_state_machine import (
    InterceptionAction,
    InterceptionGateConfig,
    InterceptionPhase,
    InterceptionStateMachine,
)


def _safe_step(machine, now_s, **overrides):
    values = {
        'hover_ready': True,
        'landed': False,
        'armed': True,
        'offboard': True,
        'controller_valid': True,
        'telemetry_age_s': 0.01,
        'command_age_s': 0.01,
        'speed_m_s': 0.0,
        'tilt_rad': 0.0,
        'barrier_margin': 0.2,
    }
    values.update(overrides)
    return machine.step(now_s=now_s, **values)


def test_disabled_gate_never_authorizes_commands() -> None:
    """The default configuration has no command-producing transition."""
    machine = InterceptionStateMachine(InterceptionGateConfig())
    for now_s in (0.0, 1.0, 100.0):
        assert _safe_step(machine, now_s) == ()
        assert machine.phase == InterceptionPhase.DISABLED
        assert not machine.command_authorized


def test_enabled_gate_requires_hover_then_prestreams() -> None:
    """Rate commands begin only after hover readiness and prestream timing."""
    machine = InterceptionStateMachine(
        InterceptionGateConfig(enable_control=True)
    )
    actions = _safe_step(machine, 0.0, hover_ready=False)
    assert actions == (InterceptionAction.HOLD_POSITION,)
    actions = _safe_step(machine, 1.0)
    assert actions == (InterceptionAction.STREAM_RATE_SETPOINT,)
    assert machine.phase == InterceptionPhase.PRESTREAM
    actions = _safe_step(machine, 2.0)
    assert actions == (InterceptionAction.STREAM_RATE_SETPOINT,)
    assert machine.phase == InterceptionPhase.ACTIVE


def test_gate_has_no_range_input() -> None:
    """The flight-state transition API cannot receive target range."""
    parameters = inspect.signature(InterceptionStateMachine.step).parameters
    assert 'distance_m' not in parameters


def test_detection_exits_then_stabilizes_after_configured_duration() -> None:
    """A visual impact event follows the explicit post-impact phases."""
    machine = InterceptionStateMachine(
        InterceptionGateConfig(
            enable_control=True,
            post_hit_coast_duration_s=2.0,
        )
    )
    _safe_step(machine, 0.0)
    _safe_step(machine, 1.0)
    actions = _safe_step(
        machine,
        1.1,
        controller_valid=False,
        interception_detected=True,
    )
    assert actions == (InterceptionAction.STREAM_EXIT_SETPOINT,)
    assert machine.phase == InterceptionPhase.IMPACT_DETECTED
    assert machine.terminal_reason == 'interception_detected'
    assert _safe_step(machine, 1.101) == (
        InterceptionAction.STREAM_EXIT_SETPOINT,
    )
    assert machine.phase == InterceptionPhase.EXIT_INTERCEPTION
    assert _safe_step(machine, 3.100) == (
        InterceptionAction.STREAM_EXIT_SETPOINT,
    )
    assert _safe_step(machine, 3.101) == (
        InterceptionAction.STREAM_STABILIZE_SETPOINT,
    )
    assert machine.phase == InterceptionPhase.STABILIZE


def test_recovery_settles_into_continuous_hover() -> None:
    machine = InterceptionStateMachine(
        InterceptionGateConfig(
            enable_control=True,
            recovery_settle_time_s=0.5,
        )
    )
    _safe_step(machine, 0.0)
    _safe_step(machine, 1.0)
    _safe_step(machine, 1.1, interception_detected=True)
    _safe_step(machine, 1.11)
    _safe_step(machine, 3.11)
    actions = _safe_step(machine, 3.2, recovery_ready=True)
    assert actions == (InterceptionAction.STREAM_STABILIZE_SETPOINT,)
    actions = _safe_step(machine, 3.7, recovery_ready=True)
    assert actions == (InterceptionAction.STREAM_HOVER_SETPOINT,)
    assert machine.phase == InterceptionPhase.HOVER
    assert _safe_step(machine, 5.0) == (
        InterceptionAction.STREAM_HOVER_SETPOINT,
    )


def test_recovery_timeout_aborts_to_land() -> None:
    machine = InterceptionStateMachine(
        InterceptionGateConfig(
            enable_control=True,
            recovery_timeout_s=2.0,
        )
    )
    _safe_step(machine, 0.0)
    _safe_step(machine, 1.0)
    _safe_step(machine, 1.1, interception_detected=True)
    _safe_step(machine, 1.11)
    _safe_step(machine, 3.12)
    actions = _safe_step(machine, 5.13)
    assert actions == (InterceptionAction.REQUEST_LAND,)
    assert machine.phase == InterceptionPhase.ABORT
    assert machine.terminal_reason == 'interception_detected'


def test_active_safety_violation_aborts_to_land() -> None:
    """A stale controller command removes authorization without a grace gap."""
    machine = InterceptionStateMachine(
        InterceptionGateConfig(enable_control=True)
    )
    _safe_step(machine, 0.0)
    _safe_step(machine, 1.0)
    actions = _safe_step(machine, 1.1, command_age_s=0.11)
    assert actions == (InterceptionAction.REQUEST_LAND,)
    assert machine.phase == InterceptionPhase.ABORT
    assert machine.terminal_reason == 'command_timeout'


def test_barrier_violation_aborts_instead_of_raising() -> None:
    """A negative barrier margin is a controlled flight abort condition."""
    machine = InterceptionStateMachine(
        InterceptionGateConfig(enable_control=True)
    )
    _safe_step(machine, 0.0)
    actions = _safe_step(machine, 0.1, barrier_margin=-0.01)
    assert actions == (InterceptionAction.REQUEST_LAND,)
    assert machine.phase == InterceptionPhase.ABORT
    assert machine.terminal_reason == 'barrier_margin'


def test_every_active_flight_safety_gate_requests_landing() -> None:
    """All configured runtime failures revoke rate control immediately."""
    failures = (
        ({'landed': True}, 'unexpected_landed'),
        ({'armed': False}, 'unexpected_disarmed'),
        ({'offboard': False}, 'offboard_lost'),
        ({'controller_valid': False}, 'controller_invalid'),
        ({'telemetry_age_s': 0.21}, 'telemetry_timeout'),
        ({'speed_m_s': 2.01}, 'speed_limit'),
        ({'tilt_rad': 0.36}, 'tilt_limit'),
    )
    for override, reason in failures:
        machine = InterceptionStateMachine(
            InterceptionGateConfig(enable_control=True)
        )
        _safe_step(machine, 0.0)
        _safe_step(machine, 1.0)
        actions = _safe_step(machine, 1.1, **override)
        assert actions == (InterceptionAction.REQUEST_LAND,)
        assert machine.phase == InterceptionPhase.ABORT
        assert machine.terminal_reason == reason


def test_mission_timeout_ends_normally_by_requesting_land() -> None:
    """A miss cannot leave rate control running beyond the bounded trial."""
    machine = InterceptionStateMachine(
        InterceptionGateConfig(enable_control=True, mission_timeout_s=3.0)
    )
    _safe_step(machine, 0.0)
    _safe_step(machine, 1.0)
    actions = _safe_step(machine, 4.0)
    assert actions == (InterceptionAction.REQUEST_LAND,)
    assert machine.phase == InterceptionPhase.LAND
    assert machine.terminal_reason == 'mission_timeout'
