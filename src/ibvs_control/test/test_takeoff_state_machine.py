import math

from ibvs_control.takeoff_state_machine import (
    FlightAction,
    FlightPhase,
    TakeoffConfig,
    TakeoffStateMachine,
)


def update_telemetry(
    machine: TakeoffStateMachine,
    now_s: float,
    *,
    armed: bool = False,
    offboard: bool = False,
    auto_land_active: bool = False,
    preflight_ok: bool = True,
    failsafe: bool = False,
    z: float = 0.0,
    vz: float = 0.0,
    tilt_rad: float = 0.0,
    landed: bool = True,
) -> None:
    """Provide one coherent set of simulated PX4 feedback."""
    machine.update_status(
        armed=armed,
        offboard=offboard,
        auto_land_active=auto_land_active,
        preflight_ok=preflight_ok,
        failsafe=failsafe,
        now_s=now_s,
    )
    machine.update_position(
        x=0.0,
        y=0.0,
        z=z,
        vz=vz,
        valid=True,
        now_s=now_s,
    )
    machine.update_attitude(
        tilt_rad=tilt_rad,
        valid=True,
        now_s=now_s,
    )
    machine.update_landed(landed)


def make_machine(**overrides) -> TakeoffStateMachine:
    """Create a state machine with short deterministic test durations."""
    values = {
        'prestream_duration_s': 1.0,
        'command_retry_period_s': 0.5,
        'altitude_settle_time_s': 0.5,
        'hold_duration_s': 1.0,
        'telemetry_timeout_s': 0.5,
    }
    values.update(overrides)
    return TakeoffStateMachine(TakeoffConfig(**values))


def advance_to_takeoff(machine: TakeoffStateMachine) -> None:
    """Advance through feedback-confirmed Offboard and arming."""
    update_telemetry(machine, 0.0)
    assert machine.step(0.0) == (
        FlightAction.STREAM_POSITION_SETPOINT,
    )
    assert machine.phase == FlightPhase.PRESTREAM

    update_telemetry(machine, 1.0)
    actions = machine.step(1.0)
    assert FlightAction.REQUEST_OFFBOARD in actions
    assert machine.phase == FlightPhase.OFFBOARD

    update_telemetry(machine, 1.1, offboard=True)
    actions = machine.step(1.1)
    assert FlightAction.REQUEST_ARM in actions
    assert machine.phase == FlightPhase.ARM

    update_telemetry(
        machine,
        1.2,
        armed=True,
        offboard=True,
        landed=False,
    )
    machine.step(1.2)
    assert machine.phase == FlightPhase.TAKEOFF
    assert machine.armed_by_us


def test_feedback_drives_complete_takeoff_hold_and_land() -> None:
    """The nominal mission completes only after PX4 reports landed/disarmed."""
    machine = make_machine()
    advance_to_takeoff(machine)

    update_telemetry(
        machine,
        1.3,
        armed=True,
        offboard=True,
        z=-2.0,
        landed=False,
    )
    machine.step(1.3)
    assert machine.phase == FlightPhase.TAKEOFF

    update_telemetry(
        machine,
        1.8,
        armed=True,
        offboard=True,
        z=-2.0,
        landed=False,
    )
    machine.step(1.8)
    assert machine.phase == FlightPhase.HOLD

    update_telemetry(
        machine,
        2.81,
        armed=True,
        offboard=True,
        z=-2.0,
        landed=False,
    )
    actions = machine.step(2.81)
    assert machine.phase == FlightPhase.LAND
    assert FlightAction.REQUEST_LAND in actions

    update_telemetry(
        machine,
        2.9,
        armed=True,
        auto_land_active=True,
        z=-1.5,
        landed=False,
    )
    actions = machine.step(2.9)
    assert FlightAction.STREAM_POSITION_SETPOINT not in actions

    update_telemetry(machine, 3.0, landed=True)
    assert machine.step(3.0) == ()
    assert machine.phase == FlightPhase.COMPLETE


def test_initialization_times_out_without_feedback() -> None:
    """A missing PX4 status causes an inert ABORT and never sends arm."""
    machine = make_machine(init_timeout_s=1.0)
    actions = machine.step(1.0)
    assert actions == ()
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'vehicle_status_not_received'
    assert not machine.armed_by_us


def test_initial_clock_sample_starts_initialization_timeout() -> None:
    """The first nonzero simulation time must not cause an immediate ABORT."""
    config = TakeoffConfig(init_timeout_s=1.0)
    machine = TakeoffStateMachine(config, start_time_s=None)

    assert machine.step(123.0) == ()
    assert machine.phase == FlightPhase.INIT
    assert machine.step(123.99) == ()
    assert machine.phase == FlightPhase.INIT

    assert machine.step(124.0) == ()
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'vehicle_status_not_received'


def test_status_timeout_after_arm_requests_landing() -> None:
    """Loss of commander feedback after our arm request enters safe landing."""
    machine = make_machine()
    advance_to_takeoff(machine)

    actions = machine.step(1.8)
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'vehicle_status_timeout'
    assert FlightAction.REQUEST_LAND in actions


def test_offboard_loss_during_takeoff_aborts() -> None:
    """Flight cannot continue after PX4 leaves Offboard unexpectedly."""
    machine = make_machine()
    advance_to_takeoff(machine)
    update_telemetry(machine, 1.3, armed=True, landed=False)

    actions = machine.step(1.3)
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'offboard_lost'
    assert FlightAction.REQUEST_LAND in actions


def test_arm_waits_for_preflight_checks_after_offboard() -> None:
    """No arm command is sent until PX4 accepts arming in Offboard mode."""
    machine = make_machine(arm_timeout_s=1.0)
    update_telemetry(machine, 0.0, preflight_ok=False)
    machine.step(0.0)
    assert machine.phase == FlightPhase.PRESTREAM

    update_telemetry(machine, 1.0, preflight_ok=False)
    actions = machine.step(1.0)
    assert FlightAction.REQUEST_OFFBOARD in actions

    update_telemetry(
        machine,
        1.1,
        offboard=True,
        preflight_ok=False,
    )
    actions = machine.step(1.1)
    assert machine.phase == FlightPhase.ARM
    assert FlightAction.REQUEST_ARM not in actions

    update_telemetry(
        machine,
        1.2,
        offboard=True,
        preflight_ok=True,
    )
    actions = machine.step(1.2)
    assert FlightAction.REQUEST_ARM in actions


def test_failed_preflight_checks_time_out_without_arming() -> None:
    """Persistent failed checks abort rather than bypass PX4 arm safety."""
    machine = make_machine(arm_timeout_s=1.0)
    update_telemetry(machine, 0.0, preflight_ok=False)
    machine.step(0.0)
    update_telemetry(machine, 1.0, preflight_ok=False)
    machine.step(1.0)
    update_telemetry(
        machine,
        1.1,
        offboard=True,
        preflight_ok=False,
    )
    machine.step(1.1)
    update_telemetry(
        machine,
        2.1,
        offboard=True,
        preflight_ok=False,
    )
    actions = machine.step(2.1)

    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'preflight_checks_failed'
    assert FlightAction.REQUEST_ARM not in actions


def test_nonfinite_position_during_flight_aborts() -> None:
    """Non-finite position is rejected before a setpoint can use it."""
    machine = make_machine()
    advance_to_takeoff(machine)
    machine.update_position(
        x=math.nan,
        y=0.0,
        z=0.0,
        vz=0.0,
        valid=True,
        now_s=1.3,
    )
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'invalid_local_position'


def test_tilt_limit_aborts_active_flight() -> None:
    """Excessive body tilt triggers ABORT and a landing request."""
    machine = make_machine(max_tilt_rad=math.radians(20.0))
    advance_to_takeoff(machine)
    update_telemetry(
        machine,
        1.3,
        armed=True,
        offboard=True,
        tilt_rad=math.radians(21.0),
        landed=False,
    )
    actions = machine.step(1.3)
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'excessive_tilt'
    assert FlightAction.REQUEST_LAND in actions


def test_horizontal_geofence_aborts_active_flight() -> None:
    """Horizontal motion beyond the P1 test area triggers ABORT."""
    machine = make_machine(max_horizontal_distance_m=1.0)
    advance_to_takeoff(machine)
    machine.update_status(
        armed=True,
        offboard=True,
        auto_land_active=False,
        preflight_ok=True,
        failsafe=False,
        now_s=1.3,
    )
    machine.update_position(
        x=1.1,
        y=0.0,
        z=-1.0,
        vz=0.0,
        valid=True,
        now_s=1.3,
    )
    machine.update_attitude(tilt_rad=0.0, valid=True, now_s=1.3)
    actions = machine.step(1.3)
    assert machine.phase == FlightPhase.ABORT
    assert machine.abort_reason == 'horizontal_geofence'
    assert FlightAction.REQUEST_LAND in actions


def test_external_sequence_can_request_land_from_hold() -> None:
    """A completed independent test can hand HOLD back to safe landing."""
    machine = make_machine(auto_land=False)
    advance_to_takeoff(machine)
    update_telemetry(
        machine,
        1.3,
        armed=True,
        offboard=True,
        z=-2.0,
        landed=False,
    )
    machine.step(1.3)
    update_telemetry(
        machine,
        1.8,
        armed=True,
        offboard=True,
        z=-2.0,
        landed=False,
    )
    machine.step(1.8)
    assert machine.phase == FlightPhase.HOLD

    assert machine.request_land(2.0)
    actions = machine.step(2.0)
    assert machine.phase == FlightPhase.LAND
    assert FlightAction.REQUEST_LAND in actions
