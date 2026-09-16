"""
Pure state machine for the P1 PX4 Offboard takeoff and landing sequence.

The state machine deliberately has no ROS dependencies.  Callbacks update its
telemetry snapshot and a periodic caller executes :meth:`step` with simulation
time.  This keeps transition and safety behavior deterministic and testable.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Optional, Tuple


class FlightPhase(str, Enum):
    """Phases of the P1 takeoff mission."""

    INIT = 'INIT'
    PRESTREAM = 'PRESTREAM'
    OFFBOARD = 'OFFBOARD'
    ARM = 'ARM'
    TAKEOFF = 'TAKEOFF'
    HOLD = 'HOLD'
    LAND = 'LAND'
    COMPLETE = 'COMPLETE'
    ABORT = 'ABORT'


class FlightAction(str, Enum):
    """Actions requested by one state-machine step."""

    STREAM_POSITION_SETPOINT = 'STREAM_POSITION_SETPOINT'
    REQUEST_OFFBOARD = 'REQUEST_OFFBOARD'
    REQUEST_ARM = 'REQUEST_ARM'
    REQUEST_LAND = 'REQUEST_LAND'


@dataclass(frozen=True)
class TakeoffConfig:
    """Timing, completion, and safety settings for a takeoff mission."""

    target_altitude_m: float = 2.0
    altitude_tolerance_m: float = 0.15
    vertical_speed_tolerance_m_s: float = 0.30
    altitude_settle_time_s: float = 1.0
    prestream_duration_s: float = 2.0
    command_retry_period_s: float = 1.0
    init_timeout_s: float = 15.0
    offboard_timeout_s: float = 10.0
    arm_timeout_s: float = 10.0
    takeoff_timeout_s: float = 30.0
    hold_duration_s: float = 10.0
    land_timeout_s: float = 30.0
    telemetry_timeout_s: float = 1.0
    max_horizontal_distance_m: float = 3.0
    max_relative_altitude_m: float = 4.0
    min_hold_altitude_m: float = 0.5
    max_tilt_rad: float = math.radians(35.0)
    auto_land: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` for unsafe or inconsistent settings."""
        positive_values = {
            'target_altitude_m': self.target_altitude_m,
            'altitude_tolerance_m': self.altitude_tolerance_m,
            'vertical_speed_tolerance_m_s': (
                self.vertical_speed_tolerance_m_s
            ),
            'altitude_settle_time_s': self.altitude_settle_time_s,
            'prestream_duration_s': self.prestream_duration_s,
            'command_retry_period_s': self.command_retry_period_s,
            'init_timeout_s': self.init_timeout_s,
            'offboard_timeout_s': self.offboard_timeout_s,
            'arm_timeout_s': self.arm_timeout_s,
            'takeoff_timeout_s': self.takeoff_timeout_s,
            'land_timeout_s': self.land_timeout_s,
            'telemetry_timeout_s': self.telemetry_timeout_s,
            'max_horizontal_distance_m': self.max_horizontal_distance_m,
            'max_relative_altitude_m': self.max_relative_altitude_m,
            'min_hold_altitude_m': self.min_hold_altitude_m,
            'max_tilt_rad': self.max_tilt_rad,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and greater than zero')

        if not math.isfinite(self.hold_duration_s):
            raise ValueError('hold_duration_s must be finite')
        if self.auto_land and self.hold_duration_s < 0.0:
            raise ValueError('hold_duration_s must not be negative')
        if self.max_relative_altitude_m <= self.target_altitude_m:
            raise ValueError(
                'max_relative_altitude_m must exceed target_altitude_m'
            )
        if self.min_hold_altitude_m >= self.target_altitude_m:
            raise ValueError(
                'min_hold_altitude_m must be below target_altitude_m'
            )
        if self.max_tilt_rad >= math.pi / 2.0:
            raise ValueError('max_tilt_rad must be less than pi/2')


class TakeoffStateMachine:
    """Drive a takeoff, timed hold, and landing from PX4 feedback."""

    def __init__(
        self,
        config: TakeoffConfig,
        start_time_s: Optional[float] = 0.0,
    ) -> None:
        config.validate()
        self.config = config
        self.phase = FlightPhase.INIT
        self.abort_reason = ''
        self.phase_entered_s = 0.0 if start_time_s is None else start_time_s
        self.clock_started = start_time_s is not None
        self.last_transition = (FlightPhase.INIT, FlightPhase.INIT)

        self.status_received = False
        self.position_received = False
        self.attitude_received = False
        self.last_status_s: Optional[float] = None
        self.last_position_s: Optional[float] = None
        self.last_attitude_s: Optional[float] = None

        self.armed = False
        self.armed_by_us = False
        self.offboard = False
        self.auto_land_active = False
        self.preflight_ok = False
        self.failsafe = False
        self.landed = True

        self.home_x: Optional[float] = None
        self.home_y: Optional[float] = None
        self.home_z: Optional[float] = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vz = 0.0
        self.tilt_rad = 0.0
        self.altitude_reached_s: Optional[float] = None
        self.last_action_s: Dict[FlightAction, float] = {}

    @property
    def terminal(self) -> bool:
        """Return whether the mission can no longer change phase."""
        return self.phase in (FlightPhase.COMPLETE, FlightPhase.ABORT)

    @property
    def relative_altitude_m(self) -> float:
        """Return altitude above the position latched during INIT."""
        if self.home_z is None:
            return 0.0
        return self.home_z - self.z

    @property
    def target_z_ned_m(self) -> Optional[float]:
        """Return the target down coordinate in the local NED frame."""
        if self.home_z is None:
            return None
        return self.home_z - self.config.target_altitude_m

    def update_status(
        self,
        *,
        armed: bool,
        offboard: bool,
        auto_land_active: bool,
        preflight_ok: bool,
        failsafe: bool,
        now_s: float,
    ) -> None:
        """Update the latest PX4 commander status."""
        self.status_received = True
        self.last_status_s = now_s
        self.armed = armed
        self.offboard = offboard
        self.auto_land_active = auto_land_active
        self.preflight_ok = preflight_ok
        self.failsafe = failsafe

        if self.phase == FlightPhase.ARM and armed:
            self.armed_by_us = True

    def update_position(
        self,
        *,
        x: float,
        y: float,
        z: float,
        vz: float,
        valid: bool,
        now_s: float,
    ) -> None:
        """Update local NED position and latch the initial position."""
        values = (x, y, z, vz)
        if not valid or not all(math.isfinite(value) for value in values):
            if self.phase != FlightPhase.INIT:
                self.abort('invalid_local_position', now_s)
            return

        self.position_received = True
        self.last_position_s = now_s
        self.x = x
        self.y = y
        self.z = z
        self.vz = vz

        if self.home_z is None:
            self.home_x = x
            self.home_y = y
            self.home_z = z

    def update_attitude(
        self,
        *,
        tilt_rad: float,
        valid: bool,
        now_s: float,
    ) -> None:
        """Update vehicle tilt used by the attitude safety check."""
        if not valid or not math.isfinite(tilt_rad):
            if self.phase != FlightPhase.INIT:
                self.abort('invalid_attitude', now_s)
            return
        self.attitude_received = True
        self.last_attitude_s = now_s
        self.tilt_rad = tilt_rad

    def update_landed(self, landed: bool) -> None:
        """Update the PX4 landing detector state."""
        self.landed = landed

    def abort(self, reason: str, now_s: float) -> None:
        """Enter ABORT once and retain the first failure reason."""
        if self.phase in (FlightPhase.COMPLETE, FlightPhase.ABORT):
            return
        self.abort_reason = reason
        self._transition(FlightPhase.ABORT, now_s)

    def request_land(self, now_s: float) -> bool:
        """Leave HOLD for LAND when an external test sequence is complete."""
        if self.phase != FlightPhase.HOLD:
            return False
        self._transition(FlightPhase.LAND, now_s)
        return True

    def step(self, now_s: float) -> Tuple[FlightAction, ...]:
        """Advance the mission and return actions to execute this cycle."""
        if not self.clock_started:
            self.phase_entered_s = now_s
            self.clock_started = True

        self._run_safety_checks(now_s)
        actions = []

        if self.phase == FlightPhase.INIT:
            if now_s - self.phase_entered_s >= self.config.init_timeout_s:
                self.abort(self._initialization_failure(), now_s)
            elif (
                self.status_received
                and self.position_received
                and self.attitude_received
            ):
                if self.armed:
                    self.abort('vehicle_already_armed', now_s)
                else:
                    self._transition(FlightPhase.PRESTREAM, now_s)

        if self.phase == FlightPhase.PRESTREAM:
            actions.append(FlightAction.STREAM_POSITION_SETPOINT)
            if (
                now_s - self.phase_entered_s
                >= self.config.prestream_duration_s
            ):
                self._transition(FlightPhase.OFFBOARD, now_s)

        if self.phase == FlightPhase.OFFBOARD:
            actions.append(FlightAction.STREAM_POSITION_SETPOINT)
            if self.offboard:
                self._transition(FlightPhase.ARM, now_s)
            elif (
                now_s - self.phase_entered_s
                >= self.config.offboard_timeout_s
            ):
                self.abort('offboard_timeout', now_s)
            elif self._action_due(FlightAction.REQUEST_OFFBOARD, now_s):
                actions.append(FlightAction.REQUEST_OFFBOARD)

        if self.phase == FlightPhase.ARM:
            actions.append(FlightAction.STREAM_POSITION_SETPOINT)
            if not self.offboard:
                self.abort('offboard_lost_before_arm', now_s)
            elif self.armed:
                self.armed_by_us = True
                self._transition(FlightPhase.TAKEOFF, now_s)
            elif now_s - self.phase_entered_s >= self.config.arm_timeout_s:
                reason = (
                    'arm_timeout'
                    if self.preflight_ok
                    else 'preflight_checks_failed'
                )
                self.abort(reason, now_s)
            elif (
                self.preflight_ok
                and self._action_due(FlightAction.REQUEST_ARM, now_s)
            ):
                actions.append(FlightAction.REQUEST_ARM)

        if self.phase == FlightPhase.TAKEOFF:
            actions.append(FlightAction.STREAM_POSITION_SETPOINT)
            if now_s - self.phase_entered_s >= self.config.takeoff_timeout_s:
                self.abort('takeoff_timeout', now_s)
            elif self._at_target_altitude():
                if self.altitude_reached_s is None:
                    self.altitude_reached_s = now_s
                elif (
                    now_s - self.altitude_reached_s
                    >= self.config.altitude_settle_time_s
                ):
                    self._transition(FlightPhase.HOLD, now_s)
            else:
                self.altitude_reached_s = None

        if self.phase == FlightPhase.HOLD:
            actions.append(FlightAction.STREAM_POSITION_SETPOINT)
            if (
                self.config.auto_land
                and now_s - self.phase_entered_s
                >= self.config.hold_duration_s
            ):
                self._transition(FlightPhase.LAND, now_s)

        if self.phase == FlightPhase.LAND:
            if not self.armed and self.landed:
                self._transition(FlightPhase.COMPLETE, now_s)
            elif now_s - self.phase_entered_s >= self.config.land_timeout_s:
                self.abort('land_timeout', now_s)
            else:
                if not self.auto_land_active:
                    actions.append(FlightAction.STREAM_POSITION_SETPOINT)
                if self._action_due(FlightAction.REQUEST_LAND, now_s):
                    actions.append(FlightAction.REQUEST_LAND)

        if self.phase == FlightPhase.ABORT:
            if (
                self.armed_by_us
                and self.armed
                and self._action_due(FlightAction.REQUEST_LAND, now_s)
            ):
                actions.append(FlightAction.REQUEST_LAND)

        return self._deduplicate(actions)

    def _run_safety_checks(self, now_s: float) -> None:
        """Apply feedback, freshness, attitude, and geofence checks."""
        if self.phase in (
            FlightPhase.INIT,
            FlightPhase.LAND,
            FlightPhase.COMPLETE,
            FlightPhase.ABORT,
        ):
            return

        if self.failsafe:
            self.abort('px4_failsafe', now_s)
            return
        if self._is_stale(self.last_status_s, now_s):
            self.abort('vehicle_status_timeout', now_s)
            return
        if self._is_stale(self.last_position_s, now_s):
            self.abort('local_position_timeout', now_s)
            return
        if self.attitude_received and self._is_stale(
            self.last_attitude_s,
            now_s,
        ):
            self.abort('vehicle_attitude_timeout', now_s)
            return
        if self.attitude_received and self.tilt_rad > self.config.max_tilt_rad:
            self.abort('excessive_tilt', now_s)
            return

        if self.home_x is None or self.home_y is None:
            return
        horizontal_distance = math.hypot(
            self.x - self.home_x,
            self.y - self.home_y,
        )
        if horizontal_distance > self.config.max_horizontal_distance_m:
            self.abort('horizontal_geofence', now_s)
            return
        if self.relative_altitude_m > self.config.max_relative_altitude_m:
            self.abort('altitude_geofence', now_s)
            return
        if (
            self.phase == FlightPhase.HOLD
            and self.relative_altitude_m < self.config.min_hold_altitude_m
        ):
            self.abort('unexpected_low_altitude', now_s)
            return

        if self.phase in (FlightPhase.TAKEOFF, FlightPhase.HOLD):
            if not self.armed:
                self.abort('unexpected_disarm', now_s)
            elif not self.offboard:
                self.abort('offboard_lost', now_s)

    def _at_target_altitude(self) -> bool:
        altitude_error = abs(
            self.config.target_altitude_m - self.relative_altitude_m
        )
        return (
            altitude_error <= self.config.altitude_tolerance_m
            and abs(self.vz) <= self.config.vertical_speed_tolerance_m_s
        )

    def _is_stale(self, timestamp_s: Optional[float], now_s: float) -> bool:
        return (
            timestamp_s is None
            or now_s - timestamp_s > self.config.telemetry_timeout_s
        )

    def _initialization_failure(self) -> str:
        if not self.status_received:
            return 'vehicle_status_not_received'
        if not self.position_received:
            return 'local_position_not_received'
        if not self.attitude_received:
            return 'vehicle_attitude_not_received'
        return 'initialization_timeout'

    def _action_due(self, action: FlightAction, now_s: float) -> bool:
        last_time = self.last_action_s.get(action)
        if (
            last_time is None
            or now_s - last_time >= self.config.command_retry_period_s
        ):
            self.last_action_s[action] = now_s
            return True
        return False

    def _transition(self, phase: FlightPhase, now_s: float) -> None:
        previous = self.phase
        self.phase = phase
        self.phase_entered_s = now_s
        self.last_transition = (previous, phase)

    @staticmethod
    def _deduplicate(actions) -> Tuple[FlightAction, ...]:
        return tuple(dict.fromkeys(actions))
