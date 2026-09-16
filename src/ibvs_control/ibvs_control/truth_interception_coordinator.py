"""Integrated, hard-disabled-by-default P2 truth interception coordinator."""

import math
from pathlib import Path
from typing import Optional

from interception_interfaces.msg import ControlDebug, RelativeState
import numpy as np
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
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
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool

from ibvs_control.controller_shadow import _assign_vector, _message_vector
from ibvs_control.frames import px4_quaternion_to_enu_flu_rotation
from ibvs_control.interception_state_machine import (
    InterceptionAction,
    InterceptionGateConfig,
    InterceptionPhase,
    InterceptionStateMachine,
    validate_activation_interlock,
)
from ibvs_control.offboard_takeoff import OffboardTakeoff
from ibvs_control.p1_trial_report import sanitize_trial_id
from ibvs_control.p2_trial_report import (
    build_p2_trial_report,
    utc_now_iso,
    write_p2_trial_report,
)
from ibvs_control.px4_command_adapter import (
    Px4RateThrustCommand,
    adapt_rate_thrust_command,
)
from ibvs_control.so3_controller import (
    OuterLoopConfig,
    OuterLoopResult,
    attitude_rate_feedback,
    combine_and_saturate_rates,
    compute_outer_loop,
    low_pass_rates,
)
from ibvs_control.takeoff_state_machine import FlightPhase
from ibvs_control.thrust_mapping import ThrustMappingConfig


class TruthInterceptionCoordinator(OffboardTakeoff):
    """Own takeoff, truth control, safety termination, and landing."""

    def __init__(self) -> None:
        super().__init__(
            node_name='truth_interception_coordinator',
            auto_land_default=False,
            command_output_default=False,
        )
        self._declare_interception_parameters()
        if self.command_output_enabled:
            raise RuntimeError(
                'truth closed-loop command output is retired; use the '
                'vision_interception_coordinator'
            )
        self._validate_activation_interlock()

        safe_angle_deg = self._float_parameter('safe_los_angle_deg')
        self.k_b = 1.0 - math.cos(math.radians(safe_angle_deg))
        self.outer_config = OuterLoopConfig(
            k1=self._float_parameter('k1'),
            k2=self._float_parameter('k2'),
            k_b=self.k_b,
            mass_kg=self._float_parameter('mass_kg'),
            thrust_max_n=self._float_parameter('thrust_max_n'),
            max_command_tilt_rad=math.radians(
                self._float_parameter('max_command_tilt_deg')
            ),
        )
        self.outer_config.validate()
        self.omega_limit_rad_s = self._float_parameter(
            'omega_limit_rad_s'
        )
        self.yaw_rate_limit_rad_s = self._float_parameter(
            'yaw_rate_limit_rad_s'
        )
        if self.yaw_rate_limit_rad_s > self.omega_limit_rad_s:
            raise ValueError(
                'yaw_rate_limit_rad_s must not exceed omega_limit_rad_s'
            )
        self.rate_slew_limit_rad_s2 = self._float_parameter(
            'rate_slew_limit_rad_s2'
        )
        self.rate_filter_time_constant_s = self._float_parameter(
            'rate_filter_time_constant_s'
        )
        self.attitude_rate_gain = self._float_parameter(
            'attitude_rate_gain'
        )
        if self.rate_filter_time_constant_s < 0.0:
            raise ValueError(
                'rate_filter_time_constant_s must be nonnegative'
            )
        if self.attitude_rate_gain <= 0.0:
            raise ValueError('attitude_rate_gain must be positive')
        self.attitude_feedback_yaw_enabled = self._bool_parameter(
            'attitude_feedback_yaw_enabled'
        )
        self.mapping = ThrustMappingConfig(
            mass_kg=self.outer_config.mass_kg,
            hover_thrust_normalized=self._float_parameter(
                'hover_thrust_normalized'
            ),
        )
        self.mapping.validate()
        self.camera_axis_b = self._unit_parameter('camera_axis_b')
        self.terminal_attack_distance_m = self._float_parameter(
            'terminal_attack_distance_m'
        )
        self.terminal_attack_acceleration_m_s2 = self._float_parameter(
            'terminal_attack_acceleration_m_s2'
        )
        if self.terminal_attack_distance_m <= 0.0:
            raise ValueError('terminal_attack_distance_m must be positive')
        if self.terminal_attack_acceleration_m_s2 < 0.0:
            raise ValueError(
                'terminal_attack_acceleration_m_s2 must be nonnegative'
            )
        self.gate = InterceptionStateMachine(
            InterceptionGateConfig(
                enable_control=self.command_output_enabled,
                prestream_duration_s=self._float_parameter(
                    'interception_prestream_duration_s'
                ),
                mission_timeout_s=self._float_parameter(
                    'mission_timeout_s'
                ),
                telemetry_timeout_s=self._float_parameter(
                    'interception_telemetry_timeout_s'
                ),
                command_timeout_s=self._float_parameter(
                    'command_timeout_s'
                ),
                hit_radius_m=self._float_parameter('hit_radius_m'),
                speed_limit_m_s=self._float_parameter('speed_limit_m_s'),
                tilt_limit_rad=math.radians(
                    self._float_parameter('interception_tilt_limit_deg')
                ),
                minimum_barrier_margin=self._float_parameter(
                    'minimum_barrier_margin'
                ),
                post_hit_coast_duration_s=self._float_parameter(
                    'post_hit_coast_duration_s'
                ),
                recovery_settle_time_s=self._float_parameter(
                    'recovery_settle_time_s'
                ),
                recovery_timeout_s=self._float_parameter(
                    'recovery_timeout_s'
                ),
            )
        )
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
        self.finished = False
        self.result_written = False
        self.transitions = []
        self.last_flight_phase = self.state_machine.phase
        self.last_gate_phase = self.gate.phase
        self.initial_distance_m: Optional[float] = None
        self.minimum_distance_m = math.inf
        self.final_distance_m = math.inf
        self.max_speed_m_s = 0.0
        self.max_tilt_deg = 0.0
        self.minimum_barrier_margin = math.inf
        self.active_sample_count = 0
        self.rate_saturation_count = 0
        self.thrust_saturation_count = 0
        self.mapping_saturation_count = 0
        self.rate_saturated = False
        self.filtered_omega_b = np.zeros(3)
        self.last_rate_update_ns: Optional[int] = None

        self.relative_state: Optional[RelativeState] = None
        self.vehicle_odometry: Optional[VehicleOdometry] = None
        self.relative_received_ns: Optional[int] = None
        self.odometry_received_ns: Optional[int] = None
        self.outer_result: Optional[OuterLoopResult] = None
        self.px4_command: Optional[Px4RateThrustCommand] = None
        self.control_reason = 'waiting_for_telemetry'
        self.command_computed_ns: Optional[int] = None
        self.vehicle_speed_m_s = 1e9
        self.vehicle_velocity_ned: Optional[np.ndarray] = None
        self.vehicle_heading_ned_rad: Optional[float] = None
        self.coast_velocity_ned: Optional[np.ndarray] = None
        self.coast_yaw_ned_rad: Optional[float] = None
        self.last_outer_ns: Optional[int] = None
        self.contact_detected = False
        self.target_green_confirmed = False
        self.recovery_hold_x_ned: Optional[float] = None
        self.recovery_hold_y_ned: Optional[float] = None
        self.recovery_target_z_ned: Optional[float] = None
        self.outer_period_ns = int(
            1e9 / self._float_parameter('outer_loop_hz')
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.debug_pub = self.create_publisher(
            ControlDebug,
            str(self.get_parameter('debug_topic').value),
            10,
        )
        self.create_subscription(
            RelativeState,
            str(self.get_parameter('relative_state_topic').value),
            self._relative_callback,
            qos,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter('topics.vehicle_odometry').value),
            self._odometry_callback,
            qos,
        )
        self.create_subscription(
            Contacts,
            self._topic('topics.target_contact'),
            self._contact_callback,
            qos,
        )
        confirmation_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Bool,
            self._topic('topics.target_green_confirmation'),
            self._green_confirmation_callback,
            confirmation_qos,
        )
        self.rates_pub = None
        if self.command_output_enabled:
            self.rates_pub = self.create_publisher(
                VehicleRatesSetpoint,
                str(self.get_parameter('topics.vehicle_rates_setpoint').value),
                qos,
            )
            self.get_logger().warning(
                'P2 truth interception ENABLED with all activation gates met'
            )
        else:
            self.get_logger().warning(
                'P2 coordinator disabled: debug only; no PX4 publishers created'
            )

    def _declare_interception_parameters(self) -> None:
        self.declare_parameter('confirmation_token', '')
        self.declare_parameter('trial_id', 'p2_truth_01')
        self.declare_parameter('results_directory', 'results/p2/trials')
        self.declare_parameter('trial_milestone', 'P2')
        self.declare_parameter('trial_control_mode', 'truth_closed_loop')
        self.declare_parameter('p1_pass_count', 2)
        self.declare_parameter('required_p1_pass_count', 10)
        self.declare_parameter('p1_gate_waived', False)
        self.declare_parameter('p1_gate_waiver_reason', '')
        self.declare_parameter('k1', 0.05)
        self.declare_parameter('k2', 20.0)
        self.declare_parameter('safe_los_angle_deg', 45.0)
        self.declare_parameter('mass_kg', 2.0)
        self.declare_parameter('thrust_max_n', 26.9784)
        self.declare_parameter('hover_thrust_normalized', 0.727)
        self.declare_parameter('omega_limit_rad_s', 0.5)
        self.declare_parameter('yaw_rate_limit_rad_s', 0.25)
        self.declare_parameter('rate_slew_limit_rad_s2', 2.0)
        self.declare_parameter('rate_filter_time_constant_s', 0.0)
        self.declare_parameter('attitude_rate_gain', 1.0)
        self.declare_parameter('attitude_feedback_yaw_enabled', False)
        self.declare_parameter('max_command_tilt_deg', 20.0)
        self.declare_parameter('outer_loop_hz', 50.0)
        self.declare_parameter('interception_prestream_duration_s', 1.0)
        self.declare_parameter('mission_timeout_s', 40.0)
        self.declare_parameter('interception_telemetry_timeout_s', 0.2)
        self.declare_parameter('command_timeout_s', 0.1)
        self.declare_parameter('hit_radius_m', 0.5)
        self.declare_parameter('speed_limit_m_s', 2.0)
        self.declare_parameter('interception_tilt_limit_deg', 20.0)
        self.declare_parameter('minimum_barrier_margin', 0.02)
        self.declare_parameter('post_hit_coast_duration_s', 2.0)
        self.declare_parameter('recovery_speed_tolerance_m_s', 0.35)
        self.declare_parameter('recovery_altitude_tolerance_m', 0.20)
        self.declare_parameter('recovery_settle_time_s', 1.0)
        self.declare_parameter('recovery_timeout_s', 15.0)
        self.declare_parameter('terminal_attack_distance_m', 2.5)
        self.declare_parameter('terminal_attack_acceleration_m_s2', 2.5)
        self.declare_parameter('camera_axis_b', [1.0, 0.0, 0.0])
        self.declare_parameter(
            'relative_state_topic',
            '/interception/truth/relative_state',
        )
        self.declare_parameter('expected_relative_state_source', 'truth')
        self.declare_parameter('debug_topic', '/interception/control/debug')
        self.declare_parameter(
            'topics.vehicle_rates_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
        )
        self.declare_parameter(
            'topics.vehicle_odometry',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter(
            'topics.target_contact',
            '/world/default/model/ibvs_target/link/target_link/sensor/'
            'target_contact/contact',
        )
        self.declare_parameter(
            'topics.target_green_confirmation',
            '/interception/target/green_confirmed',
        )
        self.declare_parameter(
            'target_collision_name',
            'ibvs_target::target_link::target_collision',
        )
        self.declare_parameter(
            'vehicle_collision_prefix',
            'x500_mono_cam_0::',
        )

    def _validate_activation_interlock(self) -> None:
        validate_activation_interlock(
            enabled=self.command_output_enabled,
            confirmation_token=str(
                self.get_parameter('confirmation_token').value
            ),
            p1_pass_count=self._int_parameter('p1_pass_count'),
            required_p1_pass_count=self._int_parameter(
                'required_p1_pass_count'
            ),
            p1_gate_waived=self._bool_parameter('p1_gate_waived'),
            waiver_reason=str(
                self.get_parameter('p1_gate_waiver_reason').value
            ),
        )

    def _unit_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must be a finite three-vector')
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError(f'{name} must be nonzero')
        return value / norm

    def _relative_callback(self, message: RelativeState) -> None:
        self.relative_state = message
        self.relative_received_ns = self.get_clock().now().nanoseconds

    def _odometry_callback(self, message: VehicleOdometry) -> None:
        self.vehicle_odometry = message
        self.odometry_received_ns = self.get_clock().now().nanoseconds

    def _contact_callback(self, message: Contacts) -> None:
        """Latch real Gazebo contact as the authoritative hit evidence."""
        if self.contact_detected or not _has_expected_target_contact(
            message,
            str(self.get_parameter('target_collision_name').value),
            str(self.get_parameter('vehicle_collision_prefix').value),
        ):
            return
        self._latch_coast_setpoint()
        self.contact_detected = True

    def _green_confirmation_callback(self, message: Bool) -> None:
        """Latch the visual indicator's positive red-to-green confirmation."""
        if message.data:
            self.target_green_confirmed = True

    def position_callback(self, message: VehicleLocalPosition) -> None:
        """Track vehicle speed while preserving takeoff feedback handling."""
        super().position_callback(message)
        velocity = np.asarray((message.vx, message.vy, message.vz), dtype=float)
        valid = bool(
            getattr(message, 'v_xy_valid', True)
            and message.v_z_valid
            and np.all(np.isfinite(velocity))
        )
        self.vehicle_speed_m_s = (
            float(np.linalg.norm(velocity))
            if valid
            else 1e9
        )
        if valid:
            self.vehicle_velocity_ned = velocity
        heading = float(getattr(message, 'heading', math.nan))
        if math.isfinite(heading):
            self.vehicle_heading_ned_rad = heading

    def timer_callback(self) -> None:
        """Run shadow math or the fully gated takeoff/interception mission."""
        now_ns = self.get_clock().now().nanoseconds
        if self.gate.phase in (
            InterceptionPhase.SUCCESS_COAST,
            InterceptionPhase.RECOVERY,
            InterceptionPhase.HOVER,
        ):
            # Contact has ended IBVS.  Do not keep generating pursuit commands
            # even though only recovery position setpoints would be published.
            self.outer_result = None
            self.px4_command = None
            self.control_reason = 'recovery_active'
        else:
            self._update_control(now_ns)
        self._publish_debug()
        if not self.command_output_enabled:
            return

        now_s = now_ns * 1e-9
        if self.trial_started_sim_s is None:
            self.trial_started_sim_s = now_s
        try:
            self._run_enabled_step(now_s, now_ns)
        finally:
            self._update_trial_observations()
            self._record_transitions(now_s)
            self._finish_trial_if_safe(now_s)

    def _run_enabled_step(self, now_s: float, now_ns: int) -> None:
        """Advance the integrated flight and interception state machines."""
        handling_confirmed_hit = (
            self.contact_detected
            or self.gate.phase in (
                InterceptionPhase.SUCCESS_COAST,
                InterceptionPhase.RECOVERY,
                InterceptionPhase.HOVER,
            )
        )
        flight_actions = (
            () if handling_confirmed_hit else self.state_machine.step(now_s)
        )
        self._log_phase_change()
        if (
            not handling_confirmed_hit
            and self.state_machine.phase != FlightPhase.HOLD
        ):
            self._execute_actions(flight_actions)
            if (
                self.stop_on_complete
                and self.state_machine.phase == FlightPhase.COMPLETE
            ):
                self.timer.cancel()
            return

        actions = self.gate.step(
            now_s=now_s,
            hover_ready=True,
            landed=self.state_machine.landed,
            armed=self.state_machine.armed,
            offboard=self.state_machine.offboard,
            controller_valid=self.px4_command is not None,
            telemetry_age_s=self._telemetry_age_s(now_ns),
            command_age_s=self._command_age_s(now_ns),
            speed_m_s=self.vehicle_speed_m_s,
            tilt_rad=self.state_machine.tilt_rad,
            barrier_margin=self._barrier_margin(),
            interception_detected=self.contact_detected,
            recovery_ready=self._recovery_ready(),
        )
        if InterceptionAction.HOLD_POSITION in actions:
            self._execute_actions(flight_actions)
        elif InterceptionAction.STREAM_RATE_SETPOINT in actions:
            self._publish_rate_mode()
            if self.gate.phase == InterceptionPhase.PRESTREAM:
                self._publish_hover_rate_setpoint()
            else:
                self._publish_interception_setpoint()
        elif InterceptionAction.STREAM_COAST_SETPOINT in actions:
            self._publish_coast_velocity_setpoint()
        elif InterceptionAction.STREAM_RECOVERY_SETPOINT in actions:
            self._publish_recovery_position_setpoint()
        elif InterceptionAction.STREAM_HOVER_SETPOINT in actions:
            self._publish_recovery_position_setpoint()
        elif InterceptionAction.REQUEST_LAND in actions:
            self.state_machine.request_land(now_s)
            self._execute_actions(self.state_machine.step(now_s))

    def _update_control(self, now_ns: int) -> None:
        if (
            self.last_outer_ns is None
            or now_ns - self.last_outer_ns >= self.outer_period_ns
        ):
            self.last_outer_ns = now_ns
            self._update_outer()
        outer = self.outer_result
        odometry = self.vehicle_odometry
        if outer is None or odometry is None:
            self.px4_command = None
            return
        try:
            attitude = px4_quaternion_to_enu_flu_rotation(odometry.q)
            omega2 = self.attitude_rate_gain * attitude_rate_feedback(
                outer.attitude_d_b_to_e,
                attitude,
            )
            if not self.attitude_feedback_yaw_enabled:
                omega2[2] = 0.0
            raw_omega = outer.omega1_b + omega2
            omega = combine_and_saturate_rates(
                outer.omega1_b,
                omega2,
                self.omega_limit_rad_s,
            )
            omega[2] = float(np.clip(
                omega[2],
                -self.yaw_rate_limit_rad_s,
                self.yaw_rate_limit_rad_s,
            ))
            omega = self._slew_limit_rates(omega, now_ns)
            self.rate_saturated = bool(
                np.linalg.norm(raw_omega) > self.omega_limit_rad_s
                or abs(raw_omega[2]) > self.yaw_rate_limit_rad_s
            )
            self.px4_command = adapt_rate_thrust_command(
                omega,
                outer.thrust_n,
                self.omega_limit_rad_s,
                self.mapping,
            )
            self.command_computed_ns = now_ns
            self.control_reason = 'ok'
        except ValueError as error:
            self.px4_command = None
            self.rate_saturated = False
            self.control_reason = str(error)

    def _slew_limit_rates(
        self,
        desired_omega_b: np.ndarray,
        now_ns: int,
    ) -> np.ndarray:
        """Bound rate changes so telemetry noise cannot make the vehicle twitch."""
        previous_ns = self.last_rate_update_ns
        self.last_rate_update_ns = now_ns
        if previous_ns is None:
            self.filtered_omega_b = desired_omega_b.copy()
            return desired_omega_b
        elapsed_s = max(0.0, (now_ns - previous_ns) * 1e-9)
        smoothed = low_pass_rates(
            self.filtered_omega_b,
            desired_omega_b,
            elapsed_s,
            self.rate_filter_time_constant_s,
        )
        maximum_change = self.rate_slew_limit_rad_s2 * elapsed_s
        delta = smoothed - self.filtered_omega_b
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > maximum_change and delta_norm > 1e-12:
            delta *= maximum_change / delta_norm
        self.filtered_omega_b = self.filtered_omega_b + delta
        return self.filtered_omega_b.copy()

    def _update_outer(self) -> None:
        relative = self.relative_state
        odometry = self.vehicle_odometry
        if relative is None or odometry is None:
            self.outer_result = None
            self.control_reason = 'waiting_for_telemetry'
            return
        expected_source = str(
            self.get_parameter('expected_relative_state_source').value
        )
        if relative.source != expected_source:
            self.outer_result = None
            self.control_reason = (
                'relative_state_source_is_not_' + expected_source
            )
            return
        try:
            attitude = px4_quaternion_to_enu_flu_rotation(odometry.q)
            p_r = _message_vector(relative.p_r)
            los = _message_vector(relative.los)
            attack_acceleration = _terminal_attack_acceleration(
                los,
                float(np.linalg.norm(p_r)),
                self.terminal_attack_distance_m,
                self.terminal_attack_acceleration_m_s2,
            )
            self.outer_result = compute_outer_loop(
                p_r,
                _message_vector(relative.v_r),
                los,
                attitude @ self.camera_axis_b,
                attitude,
                self.outer_config,
                target_acceleration_e=attack_acceleration,
            )
        except ValueError as error:
            self.outer_result = None
            self.control_reason = str(error)

    def _publish_debug(self) -> None:
        message = ControlDebug()
        message.stamp = self.get_clock().now().to_msg()
        outer = self.outer_result
        command = self.px4_command
        if outer is None or command is None:
            message.valid = False
            message.reason = self.control_reason
            self.debug_pub.publish(message)
            return
        attitude = px4_quaternion_to_enu_flu_rotation(self.vehicle_odometry.q)
        omega2 = self.attitude_rate_gain * attitude_rate_feedback(
            outer.attitude_d_b_to_e,
            attitude,
        )
        if not self.attitude_feedback_yaw_enabled:
            omega2[2] = 0.0
        omega = self.filtered_omega_b.copy()
        message.z1 = outer.z1
        _assign_vector(message.z2, outer.z2_e)
        _assign_vector(message.acceleration_d, outer.acceleration_d_e)
        message.attitude_d = tuple(outer.attitude_d_b_to_e.reshape(9))
        _assign_vector(message.omega1_b, outer.omega1_b)
        _assign_vector(message.omega2_b, omega2)
        _assign_vector(message.omega_d_b, omega)
        message.thrust_n = outer.thrust_n
        message.thrust_normalized = -command.thrust_body[2]
        message.thrust_saturated = outer.thrust_saturated
        message.thrust_mapping_saturated = command.thrust_saturated
        message.rate_saturated = bool(
            np.linalg.norm(outer.omega1_b + omega2)
            > self.omega_limit_rad_s
            or abs((outer.omega1_b + omega2)[2])
            > self.yaw_rate_limit_rad_s
        )
        message.valid = True
        message.reason = 'ok'
        self.debug_pub.publish(message)

    def _publish_rate_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.body_rate = True
        self.offboard_pub.publish(message)

    def _latch_recovery_setpoint(self) -> None:
        """Freeze impact x/y and choose a never-downward NED height."""
        if self.recovery_target_z_ned is not None:
            return
        configured_z = self.state_machine.target_z_ned_m
        if configured_z is None:
            return
        self.recovery_hold_x_ned = self.state_machine.x
        self.recovery_hold_y_ned = self.state_machine.y
        self.recovery_target_z_ned = _recovery_target_z_ned(
            self.state_machine.z,
            configured_z,
        )

    def _latch_coast_setpoint(self) -> None:
        """Freeze the pre-impact NED velocity vector and heading once."""
        if self.coast_velocity_ned is not None:
            return
        velocity = self.vehicle_velocity_ned
        if velocity is None or not np.all(np.isfinite(velocity)):
            velocity = np.zeros(3)
        self.coast_velocity_ned = velocity.copy()
        heading = self.vehicle_heading_ned_rad
        self.coast_yaw_ned_rad = (
            heading if heading is not None else self.target_yaw_rad
        )

    def _publish_coast_velocity_setpoint(self) -> None:
        """Hold impact velocity and heading for the configured two seconds."""
        self._latch_coast_setpoint()
        mode = OffboardControlMode()
        mode.timestamp = self._timestamp_us()
        mode.velocity = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = [math.nan, math.nan, math.nan]
        setpoint.velocity = list(self.coast_velocity_ned)
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = self.coast_yaw_ned_rad
        setpoint.yawspeed = math.nan
        self.trajectory_pub.publish(setpoint)

    def _publish_recovery_position_setpoint(self) -> None:
        """Brake, climb if needed, and hover while retaining PX4 Offboard."""
        self._latch_recovery_setpoint()
        if (
            self.recovery_hold_x_ned is None
            or self.recovery_hold_y_ned is None
            or self.recovery_target_z_ned is None
        ):
            return
        mode = OffboardControlMode()
        mode.timestamp = self._timestamp_us()
        mode.position = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = [
            self.recovery_hold_x_ned,
            self.recovery_hold_y_ned,
            self.recovery_target_z_ned,
        ]
        setpoint.velocity = [math.nan, math.nan, math.nan]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = self.target_yaw_rad
        setpoint.yawspeed = math.nan
        self.trajectory_pub.publish(setpoint)

    def _recovery_ready(self) -> bool:
        """Check braking and altitude recovery using PX4 local NED state."""
        target_z = self.recovery_target_z_ned
        if target_z is None:
            return False
        speed_tolerance = self._float_parameter(
            'recovery_speed_tolerance_m_s'
        )
        altitude_tolerance = self._float_parameter(
            'recovery_altitude_tolerance_m'
        )
        return (
            self.vehicle_speed_m_s <= speed_tolerance
            and abs(self.state_machine.z - target_z) <= altitude_tolerance
        )

    def _publish_hover_rate_setpoint(self) -> None:
        hover_n = self.outer_config.mass_kg * self.outer_config.gravity_m_s2
        command = adapt_rate_thrust_command(
            (0.0, 0.0, 0.0),
            hover_n,
            self.omega_limit_rad_s,
            self.mapping,
        )
        self._publish_px4_command(command)

    def _publish_interception_setpoint(self) -> None:
        if self.px4_command is not None:
            self._publish_px4_command(self.px4_command)

    def _publish_px4_command(self, command: Px4RateThrustCommand) -> None:
        message = VehicleRatesSetpoint()
        message.timestamp = self._timestamp_us()
        message.roll, message.pitch, message.yaw = command.rates_frd_rad_s
        message.thrust_body = list(command.thrust_body)
        message.reset_integral = False
        self.rates_pub.publish(message)

    def _telemetry_age_s(self, now_ns: int) -> float:
        if self.relative_received_ns is None or self.odometry_received_ns is None:
            return 1e9
        return max(
            now_ns - self.relative_received_ns,
            now_ns - self.odometry_received_ns,
        ) * 1e-9

    def _command_age_s(self, now_ns: int) -> float:
        if self.command_computed_ns is None:
            return 1e9
        return (now_ns - self.command_computed_ns) * 1e-9

    def _distance_m(self) -> float:
        if self.relative_state is None:
            return 1e9
        return _relative_distance(self.relative_state)

    def _barrier_margin(self) -> float:
        if self.outer_result is None:
            return -1e9
        return self.k_b - abs(self.outer_result.z1)

    def _update_trial_observations(self) -> None:
        relative = self.relative_state
        if relative is not None:
            distance = _relative_distance(relative)
            if self.initial_distance_m is None:
                self.initial_distance_m = distance
            self.minimum_distance_m = min(self.minimum_distance_m, distance)
            self.final_distance_m = distance
        if (
            math.isfinite(self.vehicle_speed_m_s)
            and self.vehicle_speed_m_s < 1e8
        ):
            self.max_speed_m_s = max(
                self.max_speed_m_s,
                self.vehicle_speed_m_s,
            )
        self.max_tilt_deg = max(
            self.max_tilt_deg,
            math.degrees(self.state_machine.tilt_rad),
        )
        if (
            self.gate.phase == InterceptionPhase.ACTIVE
            and self.outer_result is not None
            and self.px4_command is not None
        ):
            margin = self._barrier_margin()
            if math.isfinite(margin):
                self.minimum_barrier_margin = min(
                    self.minimum_barrier_margin,
                    margin,
                )
            self.active_sample_count += 1
            self.rate_saturation_count += int(self.rate_saturated)
            self.thrust_saturation_count += int(
                self.outer_result.thrust_saturated
            )
            self.mapping_saturation_count += int(
                self.px4_command.thrust_saturated
            )

    def _record_transitions(self, now_s: float) -> None:
        flight_phase = self.state_machine.phase
        if flight_phase != self.last_flight_phase:
            self.transitions.append(
                {
                    'machine': 'flight',
                    'from': self.last_flight_phase.value,
                    'to': flight_phase.value,
                    'sim_time_s': now_s,
                }
            )
            self.last_flight_phase = flight_phase
        gate_phase = self.gate.phase
        if gate_phase != self.last_gate_phase:
            self.transitions.append(
                {
                    'machine': 'interception',
                    'from': self.last_gate_phase.value,
                    'to': gate_phase.value,
                    'sim_time_s': now_s,
                }
            )
            self.get_logger().warning(
                f'Interception transition: {self.last_gate_phase.value} '
                f'-> {gate_phase.value}'
            )
            self.last_gate_phase = gate_phase

    def _finish_trial_if_safe(self, now_s: float) -> None:
        if self.result_written:
            return
        recovered_hover = (
            self.gate.phase == InterceptionPhase.HOVER
            and self.gate.terminal_reason == 'interception_detected'
            and self.target_green_confirmed
        )
        flight_complete = self.state_machine.phase == FlightPhase.COMPLETE
        abort_landed = (
            self.state_machine.phase == FlightPhase.ABORT
            and self.state_machine.landed
            and not self.state_machine.armed
        )
        if not recovered_hover and not flight_complete and not abort_landed:
            return
        # PASS requires both physical target contact and an applied green
        # target material; proximity alone is never success evidence.
        outcome = (
            'PASS'
            if recovered_hover
            else 'ABORT'
        )
        reason = (
            'hit_contact_green_confirmed'
            if recovered_hover
            else (
                self.state_machine.abort_reason
                or self.gate.terminal_reason
                or 'completed_without_terminal_reason'
            )
        )
        denominator = max(self.active_sample_count, 1)
        report = build_p2_trial_report(
            trial_id=self.trial_id,
            started_utc=self.trial_started_utc,
            ended_utc=utc_now_iso(),
            start_sim_time_s=(
                self.trial_started_sim_s
                if self.trial_started_sim_s is not None
                else now_s
            ),
            end_sim_time_s=now_s,
            outcome=outcome,
            reason=reason,
            gate_evidence={
                'measured_p1_passes': self._int_parameter('p1_pass_count'),
                'required_p1_passes': self._int_parameter(
                    'required_p1_pass_count'
                ),
                'p1_gate_waived': self._bool_parameter('p1_gate_waived'),
                'waiver_reason': str(
                    self.get_parameter('p1_gate_waiver_reason').value
                ),
            },
            parameters={
                'k1': self.outer_config.k1,
                'k2': self.outer_config.k2,
                'k_b': self.outer_config.k_b,
                'omega_limit_rad_s': self.omega_limit_rad_s,
                'yaw_rate_limit_rad_s': self.yaw_rate_limit_rad_s,
                'rate_slew_limit_rad_s2': self.rate_slew_limit_rad_s2,
                'terminal_attack_distance_m': self.terminal_attack_distance_m,
                'terminal_attack_acceleration_m_s2': (
                    self.terminal_attack_acceleration_m_s2
                ),
                'thrust_max_n': self.outer_config.thrust_max_n,
                'speed_limit_m_s': self.gate.config.speed_limit_m_s,
                'tilt_limit_deg': math.degrees(
                    self.gate.config.tilt_limit_rad
                ),
                'max_command_tilt_deg': math.degrees(
                    self.outer_config.max_command_tilt_rad
                ),
                'hit_radius_m': self.gate.config.hit_radius_m,
                'post_hit_coast_duration_s': (
                    self.gate.config.post_hit_coast_duration_s
                ),
            },
            metrics={
                'initial_distance_m': self.initial_distance_m,
                'd_min_m': self.minimum_distance_m,
                'final_distance_m': self.final_distance_m,
                'max_speed_m_s': self.max_speed_m_s,
                'max_tilt_deg': self.max_tilt_deg,
                'minimum_barrier_margin': self.minimum_barrier_margin,
                'physical_contact_detected': self.contact_detected,
                'target_green_confirmed': self.target_green_confirmed,
                'coast_velocity_ned_m_s': (
                    self.coast_velocity_ned.tolist()
                    if self.coast_velocity_ned is not None
                    else None
                ),
                'active_sample_count': self.active_sample_count,
                'rate_saturation_fraction': (
                    self.rate_saturation_count / denominator
                ),
                'thrust_saturation_fraction': (
                    self.thrust_saturation_count / denominator
                ),
                'mapping_saturation_fraction': (
                    self.mapping_saturation_count / denominator
                ),
            },
            transitions=self.transitions,
            milestone=str(self.get_parameter('trial_milestone').value),
            control_mode=str(
                self.get_parameter('trial_control_mode').value
            ),
        )
        path = write_p2_trial_report(report, self.results_directory)
        self.get_logger().warning(
            f'P2 trial result: {outcome}; reason={reason}; report={path}'
        )
        self.result_written = True
        # A successful trial remains alive because HOVER must keep streaming
        # Offboard mode and position setpoints.  Abort/landing trials may end.
        if not recovered_hover:
            self.finished = True
            self.timer.cancel()


def _relative_distance(message: RelativeState) -> float:
    """Compute distance from the canonical relative-position message field."""
    values = np.asarray(_message_vector(message.p_r), dtype=float)
    if not np.all(np.isfinite(values)):
        return 1e9
    return float(np.linalg.norm(values))


def _has_expected_target_contact(
    message: Contacts,
    target_collision_name: str,
    vehicle_collision_prefix: str,
) -> bool:
    """Return true only for target-to-configured-vehicle collisions."""
    for contact in message.contacts:
        names = (contact.collision1.name, contact.collision2.name)
        if target_collision_name not in names:
            continue
        other = names[1] if names[0] == target_collision_name else names[0]
        if other.startswith(vehicle_collision_prefix):
            return True
    return False


def _recovery_target_z_ned(
    current_z_ned_m: float,
    configured_z_ned_m: float,
) -> float:
    """
    Choose a recovery height that never commands downward motion.

    PX4 local position is NED: a more negative z is higher.  Taking the
    minimum climbs to the configured safe height when below it and holds the
    current height when already above it.
    """
    values = (current_z_ned_m, configured_z_ned_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('recovery NED heights must be finite')
    return min(values)


def _terminal_attack_acceleration(
    los,
    distance_m: float,
    activation_distance_m: float,
    acceleration_m_s2: float,
) -> np.ndarray:
    """Return a vector terminal command even when ROS fields yield tuples."""
    direction = np.asarray(los, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError('terminal attack LOS must be a finite three-vector')
    if distance_m <= activation_distance_m:
        return acceleration_m_s2 * direction
    return np.zeros(3)


def main(args=None) -> None:
    """Run the integrated coordinator in its configured safety mode."""
    import rclpy

    rclpy.init(args=args)
    node = TruthInterceptionCoordinator()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
