"""Pure safety gate for transitioning from hover to interception control."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple


CONTROL_CONFIRMATION_TOKEN = 'ENABLE_P2_TRUTH_CONTROL'


def validate_activation_interlock(
    *,
    enabled: bool,
    confirmation_token: str,
    p1_pass_count: int,
    required_p1_pass_count: int,
    p1_gate_waived: bool = False,
    waiver_reason: str = '',
) -> None:
    """Require an explicit token and completed P1 gate before activation."""
    if not enabled:
        return
    if confirmation_token != CONTROL_CONFIRMATION_TOKEN:
        raise ValueError('flight commands require the confirmation token')
    if required_p1_pass_count < 1:
        raise ValueError('required_p1_pass_count must be positive')
    if p1_pass_count >= required_p1_pass_count:
        return
    if p1_gate_waived and waiver_reason.strip():
        return
    if p1_gate_waived:
        raise ValueError('P1 gate waiver requires a documented reason')
    if p1_pass_count < required_p1_pass_count:
        raise ValueError(
            f'P1 gate incomplete: {p1_pass_count}/'
            f'{required_p1_pass_count} passing trials'
        )


class InterceptionPhase(str, Enum):
    """Phases controlled by the P2 interception safety gate."""

    DISABLED = 'DISABLED'
    WAIT_HOVER = 'WAIT_HOVER'
    PRESTREAM = 'PRESTREAM'
    ACTIVE = 'ACTIVE'
    IMPACT_DETECTED = 'IMPACT_DETECTED'
    EXIT_INTERCEPTION = 'EXIT_INTERCEPTION'
    STABILIZE = 'STABILIZE'
    HOVER = 'HOVER'
    LAND = 'LAND'
    COMPLETE = 'COMPLETE'
    ABORT = 'ABORT'

    # Compatibility aliases for the retired truth coordinator.
    SUCCESS_COAST = 'EXIT_INTERCEPTION'
    RECOVERY = 'STABILIZE'


class InterceptionAction(str, Enum):
    """Side effects requested by one state-machine step."""

    HOLD_POSITION = 'HOLD_POSITION'
    STREAM_RATE_SETPOINT = 'STREAM_RATE_SETPOINT'
    STREAM_EXIT_SETPOINT = 'STREAM_EXIT_SETPOINT'
    STREAM_STABILIZE_SETPOINT = 'STREAM_STABILIZE_SETPOINT'
    STREAM_HOVER_SETPOINT = 'STREAM_HOVER_SETPOINT'
    REQUEST_LAND = 'REQUEST_LAND'

    # Compatibility aliases for the retired truth coordinator.
    STREAM_COAST_SETPOINT = 'STREAM_EXIT_SETPOINT'
    STREAM_RECOVERY_SETPOINT = 'STREAM_STABILIZE_SETPOINT'


@dataclass(frozen=True)
class InterceptionGateConfig:
    """Conservative P2 activation, termination, and safety limits."""

    enable_control: bool = False
    prestream_duration_s: float = 1.0
    mission_timeout_s: float = 40.0
    telemetry_timeout_s: float = 0.2
    command_timeout_s: float = 0.1
    hit_radius_m: float = 0.5
    speed_limit_m_s: float = 2.0
    tilt_limit_rad: float = math.radians(20.0)
    minimum_barrier_margin: float = 0.02
    post_hit_coast_duration_s: float = 2.0
    recovery_settle_time_s: float = 1.0
    recovery_timeout_s: float = 15.0

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent gate parameters."""
        positive = {
            'prestream_duration_s': self.prestream_duration_s,
            'mission_timeout_s': self.mission_timeout_s,
            'telemetry_timeout_s': self.telemetry_timeout_s,
            'command_timeout_s': self.command_timeout_s,
            'hit_radius_m': self.hit_radius_m,
            'speed_limit_m_s': self.speed_limit_m_s,
            'tilt_limit_rad': self.tilt_limit_rad,
            'minimum_barrier_margin': self.minimum_barrier_margin,
            'post_hit_coast_duration_s': self.post_hit_coast_duration_s,
            'recovery_settle_time_s': self.recovery_settle_time_s,
            'recovery_timeout_s': self.recovery_timeout_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.tilt_limit_rad >= math.pi / 2.0:
            raise ValueError('tilt_limit_rad must be less than pi/2')
        if self.command_timeout_s > self.telemetry_timeout_s:
            raise ValueError(
                'command_timeout_s must not exceed telemetry_timeout_s'
            )


class InterceptionStateMachine:
    """Authorize rate control only after hover and fresh-command gates pass."""

    def __init__(self, config: InterceptionGateConfig) -> None:
        config.validate()
        self.config = config
        self.phase = (
            InterceptionPhase.WAIT_HOVER
            if config.enable_control
            else InterceptionPhase.DISABLED
        )
        self.phase_entered_s: Optional[float] = None
        self.active_started_s: Optional[float] = None
        self.recovery_ready_since_s: Optional[float] = None
        self.terminal_reason = ''

    @property
    def command_authorized(self) -> bool:
        """Return true only in phases allowed to emit rate setpoints."""
        return self.phase in (
            InterceptionPhase.PRESTREAM,
            InterceptionPhase.ACTIVE,
        )

    def step(
        self,
        *,
        now_s: float,
        hover_ready: bool,
        landed: bool,
        armed: bool,
        offboard: bool,
        controller_valid: bool,
        telemetry_age_s: float,
        command_age_s: float,
        speed_m_s: float,
        tilt_rad: float,
        barrier_margin: float,
        interception_detected: bool = False,
        recovery_ready: bool = False,
    ) -> Tuple[InterceptionAction, ...]:
        """Advance the gate and return explicitly authorized side effects."""
        self._validate_inputs(
            now_s,
            telemetry_age_s,
            command_age_s,
            speed_m_s,
            tilt_rad,
        )
        if not math.isfinite(barrier_margin):
            raise ValueError('barrier_margin must be finite')
        if self.phase == InterceptionPhase.DISABLED:
            return ()
        if self.phase_entered_s is None:
            self.phase_entered_s = now_s

        # The active controller supplies its own terminal observation.  For a
        # visual controller this is an image-scale event; evaluation-only
        # contact information never needs to enter this state machine.
        if self.phase == InterceptionPhase.ACTIVE and interception_detected:
            self._terminate(
                InterceptionPhase.IMPACT_DETECTED,
                'interception_detected',
                now_s,
            )

        failure = self._safety_failure(
            landed=landed,
            armed=armed,
            offboard=offboard,
            controller_valid=controller_valid,
            telemetry_age_s=telemetry_age_s,
            command_age_s=command_age_s,
            speed_m_s=speed_m_s,
            tilt_rad=tilt_rad,
            barrier_margin=barrier_margin,
        )
        if self.phase in (
            InterceptionPhase.PRESTREAM,
            InterceptionPhase.ACTIVE,
        ) and failure:
            self._terminate(InterceptionPhase.ABORT, failure, now_s)

        if self.phase == InterceptionPhase.WAIT_HOVER:
            if hover_ready and not failure:
                self._transition(InterceptionPhase.PRESTREAM, now_s)
            else:
                return (InterceptionAction.HOLD_POSITION,)

        if self.phase == InterceptionPhase.PRESTREAM:
            if now_s - self.phase_entered_s >= self.config.prestream_duration_s:
                self._transition(InterceptionPhase.ACTIVE, now_s)
                self.active_started_s = now_s
            return (InterceptionAction.STREAM_RATE_SETPOINT,)

        if self.phase == InterceptionPhase.ACTIVE:
            # Only the controller-specific observation passed as
            # interception_detected may terminate pursuit.  There is no range
            # input in this state-machine interface.
            if (
                self.active_started_s is not None
                and now_s - self.active_started_s
                >= self.config.mission_timeout_s
            ):
                self._terminate(
                    InterceptionPhase.LAND,
                    'mission_timeout',
                    now_s,
                )
            else:
                return (InterceptionAction.STREAM_RATE_SETPOINT,)

        if self.phase == InterceptionPhase.IMPACT_DETECTED:
            # Keep the first exit setpoint identical to the measured impact
            # velocity.  Holding this phase for one control tick makes the
            # requested transition sequence explicit and avoids a command gap.
            if now_s > self.phase_entered_s:
                self._transition(InterceptionPhase.EXIT_INTERCEPTION, now_s)
            return (InterceptionAction.STREAM_EXIT_SETPOINT,)

        if self.phase == InterceptionPhase.EXIT_INTERCEPTION:
            if not armed or not offboard:
                reason = (
                    'unexpected_disarmed' if not armed else 'offboard_lost'
                )
                self._terminate(InterceptionPhase.ABORT, reason, now_s)
            elif (
                now_s - self.phase_entered_s
                >= self.config.post_hit_coast_duration_s
            ):
                self._transition(InterceptionPhase.STABILIZE, now_s)
            else:
                return (InterceptionAction.STREAM_EXIT_SETPOINT,)

        if self.phase == InterceptionPhase.STABILIZE:
            if not armed or not offboard:
                reason = (
                    'unexpected_disarmed' if not armed else 'offboard_lost'
                )
                self._terminate(InterceptionPhase.ABORT, reason, now_s)
            elif (
                now_s - self.phase_entered_s
                >= self.config.recovery_timeout_s
            ):
                self._terminate(
                    InterceptionPhase.ABORT,
                    'recovery_timeout',
                    now_s,
                )
            elif recovery_ready:
                if self.recovery_ready_since_s is None:
                    self.recovery_ready_since_s = now_s
                elif (
                    now_s - self.recovery_ready_since_s
                    >= self.config.recovery_settle_time_s
                ):
                    self._transition(InterceptionPhase.HOVER, now_s)
            else:
                self.recovery_ready_since_s = None
            if self.phase == InterceptionPhase.STABILIZE:
                return (InterceptionAction.STREAM_STABILIZE_SETPOINT,)

        if self.phase == InterceptionPhase.HOVER:
            if not armed or not offboard:
                reason = (
                    'unexpected_disarmed' if not armed else 'offboard_lost'
                )
                self._terminate(InterceptionPhase.ABORT, reason, now_s)
            else:
                return (InterceptionAction.STREAM_HOVER_SETPOINT,)

        if self.phase in (
            InterceptionPhase.LAND,
            InterceptionPhase.ABORT,
        ):
            if landed and not armed:
                self._transition(InterceptionPhase.COMPLETE, now_s)
                return ()
            return (InterceptionAction.REQUEST_LAND,)
        return ()

    def _safety_failure(
        self,
        *,
        landed: bool,
        armed: bool,
        offboard: bool,
        controller_valid: bool,
        telemetry_age_s: float,
        command_age_s: float,
        speed_m_s: float,
        tilt_rad: float,
        barrier_margin: float,
    ) -> str:
        if landed:
            return 'unexpected_landed'
        if not armed:
            return 'unexpected_disarmed'
        if not offboard:
            return 'offboard_lost'
        if not controller_valid:
            return 'controller_invalid'
        if telemetry_age_s > self.config.telemetry_timeout_s:
            return 'telemetry_timeout'
        if command_age_s > self.config.command_timeout_s:
            return 'command_timeout'
        if speed_m_s > self.config.speed_limit_m_s:
            return 'speed_limit'
        if tilt_rad > self.config.tilt_limit_rad:
            return 'tilt_limit'
        if barrier_margin < self.config.minimum_barrier_margin:
            return 'barrier_margin'
        return ''

    @staticmethod
    def _validate_inputs(*values: float) -> None:
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError('state-machine numeric inputs must be nonnegative')

    def _terminate(
        self,
        phase: InterceptionPhase,
        reason: str,
        now_s: float,
    ) -> None:
        if not self.terminal_reason:
            self.terminal_reason = reason
        self._transition(phase, now_s)

    def _transition(self, phase: InterceptionPhase, now_s: float) -> None:
        self.phase = phase
        self.phase_entered_s = now_s
