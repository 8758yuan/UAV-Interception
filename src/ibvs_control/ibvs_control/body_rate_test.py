"""
Bounded PX4 body-rate and thrust direction test for the P1 milestone.

This node performs its own position-controlled takeoff, applies visible FRD
body-rate pulses with a fixed hover-thrust estimate, checks the measured
angular-rate signs, returns to hover after every pulse, and requests landing.
"""

from enum import Enum
import math
from pathlib import Path
from typing import Optional

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleLocalPosition,
    VehicleOdometry,
    VehicleRatesSetpoint,
)
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from ibvs_control.offboard_takeoff import OffboardTakeoff
from ibvs_control.p1_trial_report import (
    build_trial_report,
    sanitize_trial_id,
    utc_now_iso,
    write_trial_report,
)
from ibvs_control.rate_test_sequence import RateTestConfig, RateTestSequence
from ibvs_control.takeoff_state_machine import FlightPhase


class RateTestMode(str, Enum):
    """High-level modes around the reusable takeoff state machine."""

    TAKEOFF = 'TAKEOFF'
    RATE_TEST = 'RATE_TEST'
    RECOVER = 'RECOVER'
    LAND = 'LAND'


class BodyRateTest(OffboardTakeoff):
    """Run signed body-rate pulses after a feedback-confirmed takeoff."""

    def __init__(self) -> None:
        super().__init__(
            node_name='body_rate_test',
            auto_land_default=False,
        )
        if self.state_machine.config.auto_land:
            raise ValueError('body_rate_test requires auto_land=false')

        self.declare_parameter(
            'topics.vehicle_rates_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
        )
        self.declare_parameter(
            'topics.vehicle_odometry',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter('hover_thrust', 0.727)
        self.declare_parameter('roll_rate_rad_s', 0.30)
        self.declare_parameter('pitch_rate_rad_s', 0.30)
        self.declare_parameter('yaw_rate_rad_s', 0.45)
        self.declare_parameter('prepare_duration_s', 2.0)
        self.declare_parameter('pulse_duration_s', 0.4)
        self.declare_parameter('settle_duration_s', 2.0)
        self.declare_parameter('response_ignore_s', 0.15)
        self.declare_parameter('minimum_response_rad_s', 0.02)
        self.declare_parameter('minimum_samples', 3)
        self.declare_parameter('recovery_duration_s', 2.0)
        self.declare_parameter('recovery_position_tolerance_m', 0.20)
        self.declare_parameter(
            'recovery_horizontal_speed_tolerance_m_s',
            0.20,
        )
        self.declare_parameter('recovery_tilt_tolerance_deg', 5.0)
        self.declare_parameter('recovery_settle_time_s', 0.5)
        self.declare_parameter('recovery_timeout_s', 8.0)
        self.declare_parameter('trial_id', 'manual')
        self.declare_parameter('results_directory', 'results/p1')

        self.hover_thrust = self._float_parameter('hover_thrust')
        if not 0.1 <= self.hover_thrust <= 0.8:
            raise ValueError('hover_thrust must be within [0.1, 0.8]')
        self.recovery_duration_s = self._float_parameter(
            'recovery_duration_s'
        )
        if self.recovery_duration_s <= 0.0:
            raise ValueError('recovery_duration_s must be greater than zero')
        self.recovery_position_tolerance_m = self._positive_parameter(
            'recovery_position_tolerance_m'
        )
        self.recovery_horizontal_speed_tolerance_m_s = (
            self._positive_parameter(
                'recovery_horizontal_speed_tolerance_m_s'
            )
        )
        self.recovery_tilt_tolerance_rad = math.radians(
            self._positive_parameter('recovery_tilt_tolerance_deg')
        )
        if (
            self.recovery_tilt_tolerance_rad
            >= self.state_machine.config.max_tilt_rad
        ):
            raise ValueError(
                'recovery_tilt_tolerance_deg must be below max_tilt_deg'
            )
        self.recovery_settle_time_s = self._positive_parameter(
            'recovery_settle_time_s'
        )
        self.recovery_timeout_s = self._positive_parameter(
            'recovery_timeout_s'
        )
        if self.recovery_timeout_s <= self.recovery_settle_time_s:
            raise ValueError(
                'recovery_timeout_s must exceed recovery_settle_time_s'
            )

        sequence_config = RateTestConfig(
            roll_rate_rad_s=self._float_parameter('roll_rate_rad_s'),
            pitch_rate_rad_s=self._float_parameter('pitch_rate_rad_s'),
            yaw_rate_rad_s=self._float_parameter('yaw_rate_rad_s'),
            prepare_duration_s=self._float_parameter('prepare_duration_s'),
            pulse_duration_s=self._float_parameter('pulse_duration_s'),
            settle_duration_s=self._float_parameter('settle_duration_s'),
            response_ignore_s=self._float_parameter('response_ignore_s'),
            minimum_response_rad_s=self._float_parameter(
                'minimum_response_rad_s'
            ),
            minimum_samples=self._int_parameter('minimum_samples'),
        )
        self.sequence = RateTestSequence(sequence_config)
        self.test_mode = RateTestMode.TAKEOFF
        self.recovery_started_s = 0.0
        self.last_test_segment = ''
        self.trial_id = sanitize_trial_id(
            str(self.get_parameter('trial_id').value)
        )
        self.results_directory = str(
            self.get_parameter('results_directory').value
        )
        if not self.results_directory:
            raise ValueError('results_directory must not be empty')
        Path(self.results_directory).expanduser().resolve().mkdir(
            parents=True,
            exist_ok=True,
        )
        self.trial_started_utc = utc_now_iso()
        self.trial_started_sim_s: Optional[float] = None
        self.max_horizontal_distance_observed_m = 0.0
        self.max_relative_altitude_observed_m = 0.0
        self.max_tilt_observed_deg = 0.0
        self.horizontal_speed_m_s = math.inf
        self.recovery_gate_name = ''
        self.recovery_gate_started_s: Optional[float] = None
        self.recovery_stable_since_s: Optional[float] = None
        self.result_written = False
        self.finished = False
        self.exit_code = 0

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.rates_pub = self.create_publisher(
            VehicleRatesSetpoint,
            self._topic('topics.vehicle_rates_setpoint'),
            qos_pub,
        )
        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            self._topic('topics.vehicle_odometry'),
            self.odometry_callback,
            qos_sub,
        )
        self.get_logger().warning(
            'Body-rate test ready: the vehicle will take off, apply bounded '
            'rate pulses with hover recovery, and land automatically'
        )

    def odometry_callback(
        self,
        msg: VehicleOdometry,
    ) -> None:
        """Feed odometry's measured body-FRD rates to the evaluator."""
        if self.test_mode != RateTestMode.RATE_TEST:
            return
        try:
            self.sequence.observe(msg.angular_velocity, self._now_s())
        except ValueError:
            self.state_machine.abort(
                'invalid_angular_velocity',
                self._now_s(),
            )

    def position_callback(self, msg: VehicleLocalPosition) -> None:
        """Track horizontal speed in addition to the shared NED position."""
        super().position_callback(msg)
        velocity = (float(msg.vx), float(msg.vy))
        velocity_valid = bool(getattr(msg, 'v_xy_valid', True))
        if velocity_valid and all(math.isfinite(value) for value in velocity):
            self.horizontal_speed_m_s = math.hypot(*velocity)
        else:
            self.horizontal_speed_m_s = math.inf

    def publish_body_rate_control_mode(self) -> None:
        """Select body-rate control as the active Offboard command type."""
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        self.offboard_pub.publish(msg)

    def publish_rates_setpoint(self, rates) -> None:
        """Publish FRD angular rates with negative body-Z multicopter thrust."""
        if len(rates) != 3 or not all(math.isfinite(value) for value in rates):
            self.state_machine.abort('invalid_rate_command', self._now_s())
            return

        msg = VehicleRatesSetpoint()
        msg.timestamp = self._timestamp_us()
        msg.roll = float(rates[0])
        msg.pitch = float(rates[1])
        msg.yaw = float(rates[2])
        msg.thrust_body = [0.0, 0.0, -self.hover_thrust]
        msg.reset_integral = False
        self.rates_pub.publish(msg)

    def timer_callback(self) -> None:
        """Run takeoff, body-rate test, recovery, and landing in sequence."""
        now_s = self._now_s()
        if self.trial_started_sim_s is None:
            self.trial_started_sim_s = now_s
        try:
            self._run_test_step(now_s)
        finally:
            self._update_trial_observations()
            self._finish_trial_if_safe(now_s)

    def _run_test_step(self, now_s: float) -> None:
        """Execute one control step for the active test mode."""
        if self.test_mode == RateTestMode.TAKEOFF:
            super().timer_callback()
            if self.state_machine.phase == FlightPhase.HOLD:
                self.test_mode = RateTestMode.RATE_TEST
                self.get_logger().warning(
                    'Starting body-rate pulses using '
                    f'hover_thrust={self.hover_thrust:.3f}'
                )
            return

        actions = self.state_machine.step(now_s)
        self._log_phase_change()
        if self.state_machine.phase == FlightPhase.ABORT:
            self._execute_actions(actions)
            return

        if self.test_mode == RateTestMode.RATE_TEST:
            recovery_ready = self._position_recovery_ready(
                self.sequence.current_name,
                now_s,
            )
            rates = self.sequence.command(
                now_s,
                recovery_ready=recovery_ready,
            )
            self._log_test_segment()
            self._log_test_results()
            if any(not result.passed for result in self.sequence.results):
                failed = next(
                    result
                    for result in self.sequence.results
                    if not result.passed
                )
                self.state_machine.abort(
                    f'{failed.name.lower()}_{failed.reason}',
                    now_s,
                )
                abort_actions = self.state_machine.step(now_s)
                self._log_phase_change()
                self._execute_actions(abort_actions)
                return

            if self.sequence.complete:
                self.test_mode = RateTestMode.RECOVER
                self.recovery_started_s = now_s
                self.get_logger().info(
                    'All body-rate signs passed; recovering position control'
                )
                self.publish_offboard_control_mode()
                self.publish_trajectory_setpoint()
                return

            if self.sequence.uses_position_control:
                self.publish_offboard_control_mode()
                self.publish_trajectory_setpoint()
                return

            self.publish_body_rate_control_mode()
            self.publish_rates_setpoint(rates)
            return

        if self.test_mode == RateTestMode.RECOVER:
            self._execute_actions(actions)
            recovery_ready = self._position_recovery_ready(
                'FINAL_RECOVER',
                now_s,
            )
            minimum_elapsed = (
                now_s - self.recovery_started_s
                >= self.recovery_duration_s
            )
            if minimum_elapsed and recovery_ready:
                if self.state_machine.request_land(now_s):
                    self.test_mode = RateTestMode.LAND
                    land_actions = self.state_machine.step(now_s)
                    self._log_phase_change()
                    self._execute_actions(land_actions)
            return

        self._execute_actions(actions)

    def _positive_parameter(self, name: str) -> float:
        value = self._float_parameter(name)
        if value <= 0.0:
            raise ValueError(f'{name} must be greater than zero')
        return value

    def _position_recovery_ready(self, name: str, now_s: float) -> bool:
        """Gate the next pulse on continuously stable hover feedback."""
        is_recovery = name.endswith('_RECOVER')
        if not is_recovery:
            self.recovery_gate_name = ''
            self.recovery_gate_started_s = None
            self.recovery_stable_since_s = None
            return True

        if name != self.recovery_gate_name:
            self.recovery_gate_name = name
            self.recovery_gate_started_s = now_s
            self.recovery_stable_since_s = None
            self.get_logger().info(
                f'{name}: waiting for position and velocity to stabilize'
            )

        if (
            self.recovery_gate_started_s is not None
            and now_s - self.recovery_gate_started_s
            >= self.recovery_timeout_s
        ):
            self.state_machine.abort(
                f'{name.lower()}_timeout',
                now_s,
            )
            return False

        stable = self._hover_feedback_is_stable()
        if not stable:
            self.recovery_stable_since_s = None
            return False
        if self.recovery_stable_since_s is None:
            self.recovery_stable_since_s = now_s
            return False
        if (
            now_s - self.recovery_stable_since_s
            < self.recovery_settle_time_s
        ):
            return False

        self.get_logger().info(
            f'{name}: stable hover confirmed for '
            f'{self.recovery_settle_time_s:.2f} s'
        )
        return True

    def _hover_feedback_is_stable(self) -> bool:
        state = self.state_machine
        if state.home_x is None or state.home_y is None:
            return False
        horizontal_error = math.hypot(
            state.x - state.home_x,
            state.y - state.home_y,
        )
        altitude_error = abs(
            state.config.target_altitude_m - state.relative_altitude_m
        )
        return (
            horizontal_error <= self.recovery_position_tolerance_m
            and self.horizontal_speed_m_s
            <= self.recovery_horizontal_speed_tolerance_m_s
            and altitude_error <= state.config.altitude_tolerance_m
            and abs(state.vz) <= state.config.vertical_speed_tolerance_m_s
            and state.tilt_rad <= self.recovery_tilt_tolerance_rad
        )

    def _update_trial_observations(self) -> None:
        state = self.state_machine
        if state.home_x is not None and state.home_y is not None:
            horizontal_distance = math.hypot(
                state.x - state.home_x,
                state.y - state.home_y,
            )
            self.max_horizontal_distance_observed_m = max(
                self.max_horizontal_distance_observed_m,
                horizontal_distance,
            )
        self.max_relative_altitude_observed_m = max(
            self.max_relative_altitude_observed_m,
            state.relative_altitude_m,
        )
        self.max_tilt_observed_deg = max(
            self.max_tilt_observed_deg,
            math.degrees(state.tilt_rad),
        )

    def _finish_trial_if_safe(self, now_s: float) -> None:
        """Persist a terminal result after landing, then allow process exit."""
        if self.result_written:
            return
        phase = self.state_machine.phase
        complete = phase == FlightPhase.COMPLETE
        safe_abort = (
            phase == FlightPhase.ABORT
            and (
                not self.state_machine.armed_by_us
                or (
                    not self.state_machine.armed
                    and self.state_machine.landed
                )
            )
        )
        if not complete and not safe_abort:
            return

        passed = complete and self.sequence.passed
        outcome = 'PASS' if passed else 'ABORT'
        reason = 'ok' if passed else (
            self.state_machine.abort_reason or 'incomplete_rate_sequence'
        )
        config = self.sequence.config
        report = build_trial_report(
            trial_id=self.trial_id,
            started_utc=self.trial_started_utc,
            ended_utc=utc_now_iso(),
            start_sim_time_s=(
                now_s
                if self.trial_started_sim_s is None
                else self.trial_started_sim_s
            ),
            end_sim_time_s=now_s,
            outcome=outcome,
            reason=reason,
            terminal_phase=phase.value,
            parameters={
                'target_altitude_m': (
                    self.state_machine.config.target_altitude_m
                ),
                'hover_thrust': self.hover_thrust,
                'roll_rate_rad_s': config.roll_rate_rad_s,
                'pitch_rate_rad_s': config.pitch_rate_rad_s,
                'yaw_rate_rad_s': config.yaw_rate_rad_s,
                'pulse_duration_s': config.pulse_duration_s,
                'settle_duration_s': config.settle_duration_s,
                'recovery_position_tolerance_m': (
                    self.recovery_position_tolerance_m
                ),
                'recovery_horizontal_speed_tolerance_m_s': (
                    self.recovery_horizontal_speed_tolerance_m_s
                ),
                'recovery_tilt_tolerance_deg': math.degrees(
                    self.recovery_tilt_tolerance_rad
                ),
                'recovery_settle_time_s': self.recovery_settle_time_s,
                'recovery_timeout_s': self.recovery_timeout_s,
                'minimum_response_rad_s': (
                    config.minimum_response_rad_s
                ),
            },
            max_horizontal_distance_m=(
                self.max_horizontal_distance_observed_m
            ),
            max_relative_altitude_m=(
                self.max_relative_altitude_observed_m
            ),
            max_tilt_deg=self.max_tilt_observed_deg,
            rate_results=self.sequence.results,
        )
        try:
            report_path = write_trial_report(
                report,
                self.results_directory,
            )
        except (OSError, ValueError, KeyError) as error:
            self.get_logger().error(f'Failed to write P1 result: {error}')
            self.result_written = True
            self.exit_code = 2
            if self.stop_on_complete:
                self.finished = True
                self.timer.cancel()
            return

        self.result_written = True
        log = self.get_logger().info if passed else self.get_logger().error
        log(f'P1 trial {outcome}: {reason}; report={report_path}')
        if self.stop_on_complete:
            self.finished = True
            self.timer.cancel()

    def _log_test_results(self) -> None:
        for result in self.sequence.pop_results():
            summary = (
                f'{result.name}: command={result.commanded_rate_rad_s:+.3f} '
                f'rad/s, measured_mean={result.mean_rate_rad_s:+.3f} '
                f'rad/s, samples={result.samples}, result={result.reason}'
            )
            if result.passed:
                self.get_logger().info(summary)
            else:
                self.get_logger().error(summary)

    def _log_test_segment(self) -> None:
        """Announce each pulse and hover-recovery segment once."""
        name = self.sequence.current_name
        if name == self.last_test_segment:
            return
        self.last_test_segment = name
        if name.endswith('_RECOVER') or name == 'PREPARE':
            self.get_logger().info(f'Test segment: {name} (position hold)')
        elif name != 'COMPLETE':
            self.get_logger().warning(f'Test segment: {name} (body rate)')


def main(args=None) -> int:
    """Run the P1 body-rate direction test."""
    rclpy.init(args=args)
    node = BodyRateTest()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return node.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
