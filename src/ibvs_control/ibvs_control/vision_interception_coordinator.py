"""Integrated camera-feature-only interception coordinator."""

import math
from typing import Optional

from interception_interfaces.msg import ControlDebug, VisionFeature
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

from ibvs_control.controller_shadow import _assign_vector
from ibvs_control.frames import (
    ned_to_enu,
    px4_quaternion_to_enu_flu_rotation,
    quaternion_wxyz_to_euler,
)
from ibvs_control.interception_state_machine import (
    InterceptionAction,
    InterceptionGateConfig,
    InterceptionPhase,
    InterceptionStateMachine,
)
from ibvs_control.offboard_takeoff import OffboardTakeoff
from ibvs_control.px4_command_adapter import (
    Px4RateThrustCommand,
    adapt_rate_thrust_command,
)
from ibvs_control.post_impact_transition import smooth_stop_command
from ibvs_control.so3_controller import low_pass_rates
from ibvs_control.takeoff_state_machine import FlightPhase
from ibvs_control.thrust_mapping import ThrustMappingConfig
from ibvs_control.visual_acquisition import (
    VisualAcquisition,
    VisualAcquisitionCommand,
    VisualAcquisitionConfig,
)
from ibvs_control.visual_ibvs import (
    VisualIbvsConfig,
    VisualIbvsResult,
    compute_visual_ibvs,
    recent_close_target_lost,
    terminal_visual_ready,
)


VISION_CONTROL_CONFIRMATION_TOKEN = 'ENABLE_VISION_ONLY_CONTROL'


class VisionInterceptionCoordinator(OffboardTakeoff):
    """Take off and intercept from camera features plus onboard UAV state."""

    def __init__(self) -> None:
        super().__init__(
            node_name='vision_interception_coordinator',
            auto_land_default=False,
            command_output_default=False,
        )
        self._declare_visual_parameters()
        self._validate_activation()

        safe_angle_deg = self._float_parameter('safe_los_angle_deg')
        self.k_b = 1.0 - math.cos(math.radians(safe_angle_deg))
        self.visual_config = VisualIbvsConfig(
            mass_kg=self._float_parameter('mass_kg'),
            thrust_max_n=self._float_parameter('thrust_max_n'),
            k_b=self.k_b,
            los_rate_gain=self._float_parameter('los_rate_gain'),
            image_center_rate_gain=self._float_parameter(
                'image_center_rate_gain'
            ),
            attitude_rate_gain=self._float_parameter('attitude_rate_gain'),
            speed_gain=self._float_parameter('approach_speed_gain'),
            cruise_speed_m_s=self._float_parameter('cruise_speed_m_s'),
            terminal_speed_m_s=self._float_parameter('terminal_speed_m_s'),
            max_approach_acceleration_m_s2=self._float_parameter(
                'max_approach_acceleration_m_s2'
            ),
            max_approach_deceleration_m_s2=self._float_parameter(
                'max_approach_deceleration_m_s2'
            ),
            transverse_velocity_gain=self._float_parameter(
                'transverse_velocity_gain'
            ),
            max_transverse_acceleration_m_s2=self._float_parameter(
                'max_transverse_acceleration_m_s2'
            ),
            vertical_velocity_gain=self._float_parameter(
                'vertical_velocity_gain'
            ),
            max_vertical_correction_m_s2=self._float_parameter(
                'max_vertical_correction_m_s2'
            ),
            terminal_area_ratio=self._float_parameter(
                'terminal_area_ratio'
            ),
            full_speed_image_error=self._float_parameter(
                'full_speed_image_error'
            ),
            stop_approach_image_error=self._float_parameter(
                'stop_approach_image_error'
            ),
            max_command_tilt_rad=math.radians(
                self._float_parameter('max_command_tilt_deg')
            ),
        )
        self.visual_config.validate()
        self.acquisition = VisualAcquisition(
            VisualAcquisitionConfig(
                search_yaw_rate_rad_s=self._float_parameter(
                    'visual_search_yaw_rate_rad_s'
                ),
                align_yaw_gain_rad_s=self._float_parameter(
                    'visual_align_yaw_gain_rad_s'
                ),
                align_vertical_gain_m_s=self._float_parameter(
                    'visual_align_vertical_gain_m_s'
                ),
                center_error=self._float_parameter(
                    'visual_align_center_error'
                ),
                release_error=self._float_parameter(
                    'visual_align_release_error'
                ),
                settle_time_s=self._float_parameter(
                    'visual_align_settle_time_s'
                ),
                target_loss_timeout_s=self._float_parameter(
                    'visual_search_loss_timeout_s'
                ),
                maximum_vertical_offset_m=self._float_parameter(
                    'visual_align_max_vertical_offset_m'
                ),
            )
        )
        self.mapping = ThrustMappingConfig(
            mass_kg=self.visual_config.mass_kg,
            hover_thrust_normalized=self._float_parameter(
                'hover_thrust_normalized'
            ),
        )
        self.mapping.validate()
        self.image_area_px = self._float_parameter('image_area_px')
        self.terminal_center_error = self._float_parameter(
            'terminal_max_center_error'
        )
        self.stabilize_duration_s = self._float_parameter(
            'stabilize_duration_s'
        )
        self.stabilize_tilt_tolerance_rad = math.radians(
            self._float_parameter('stabilize_tilt_tolerance_deg')
        )
        self.impact_loss_area_ratio = self._float_parameter(
            'impact_loss_area_ratio'
        )
        self.impact_loss_center_error = self._float_parameter(
            'impact_loss_max_center_error'
        )
        self.impact_loss_timeout_s = self._float_parameter(
            'impact_loss_timeout_s'
        )
        if self.image_area_px <= 0.0:
            raise ValueError('image_area_px must be positive')
        if self.terminal_center_error <= 0.0:
            raise ValueError('terminal_max_center_error must be positive')
        if self.stabilize_duration_s <= 0.0:
            raise ValueError('stabilize_duration_s must be positive')
        if not 0.0 < self.stabilize_tilt_tolerance_rad < math.pi / 2.0:
            raise ValueError('stabilize tilt tolerance must be within (0, 90)')
        if not 0.0 < self.impact_loss_area_ratio < 1.0:
            raise ValueError('impact loss area ratio must be within (0, 1)')
        if self.impact_loss_center_error <= 0.0:
            raise ValueError('impact loss center error must be positive')
        if self.impact_loss_timeout_s <= 0.0:
            raise ValueError('impact loss timeout must be positive')

        self.omega_limit_rad_s = self._float_parameter(
            'omega_limit_rad_s'
        )
        self.yaw_rate_limit_rad_s = self._float_parameter(
            'yaw_rate_limit_rad_s'
        )
        self.rate_slew_limit_rad_s2 = self._float_parameter(
            'rate_slew_limit_rad_s2'
        )
        self.rate_filter_time_constant_s = self._float_parameter(
            'rate_filter_time_constant_s'
        )
        if not 0.0 < self.yaw_rate_limit_rad_s <= self.omega_limit_rad_s:
            raise ValueError('yaw rate limit must be within the total rate limit')
        if self.rate_slew_limit_rad_s2 <= 0.0:
            raise ValueError('rate_slew_limit_rad_s2 must be positive')
        if self.rate_filter_time_constant_s < 0.0:
            raise ValueError('rate_filter_time_constant_s must be nonnegative')

        self.gate = InterceptionStateMachine(
            InterceptionGateConfig(
                enable_control=self.command_output_enabled,
                prestream_duration_s=self._float_parameter(
                    'interception_prestream_duration_s'
                ),
                mission_timeout_s=self._float_parameter('mission_timeout_s'),
                telemetry_timeout_s=self._float_parameter(
                    'interception_telemetry_timeout_s'
                ),
                command_timeout_s=self._float_parameter('command_timeout_s'),
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

        self.feature: Optional[VisionFeature] = None
        self.last_feature: Optional[VisionFeature] = None
        self.last_valid_feature: Optional[VisionFeature] = None
        self.last_valid_feature_received_ns: Optional[int] = None
        self.target_detected = False
        self.odometry: Optional[VehicleOdometry] = None
        self.feature_received_ns: Optional[int] = None
        self.odometry_received_ns: Optional[int] = None
        self.visual_result: Optional[VisualIbvsResult] = None
        self.px4_command: Optional[Px4RateThrustCommand] = None
        self.command_computed_ns: Optional[int] = None
        self.control_reason = 'waiting_for_camera_feature'
        self.filtered_omega_b = np.zeros(3)
        self.last_rate_update_ns: Optional[int] = None
        self.last_outer_ns: Optional[int] = None
        self.outer_period_ns = int(
            1e9 / self._float_parameter('outer_loop_hz')
        )

        self.vehicle_speed_m_s = 1e9
        self.vehicle_velocity_ned: Optional[np.ndarray] = None
        self.vehicle_heading_ned_rad: Optional[float] = None
        self.terminal_started_s: Optional[float] = None
        self.visual_interception_complete = False
        self.coast_velocity_ned: Optional[np.ndarray] = None
        self.coast_yaw_ned_rad: Optional[float] = None
        self.stabilize_started_s: Optional[float] = None
        self.recovery_hold_x_ned: Optional[float] = None
        self.recovery_hold_y_ned: Optional[float] = None
        self.recovery_target_z_ned: Optional[float] = None
        self.finished = False
        self.last_gate_phase = self.gate.phase
        self.commanded_body_rates = np.full(3, math.nan)
        self.commanded_velocity_ned = np.full(3, math.nan)
        self.commanded_acceleration_ned = np.full(3, math.nan)
        self.commanded_thrust_normalized = math.nan
        self.acquisition_command: Optional[VisualAcquisitionCommand] = None
        self.last_acquisition_phase = self.acquisition.phase

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
            VisionFeature,
            str(self.get_parameter('feature_topic').value),
            self._feature_callback,
            qos,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter('topics.vehicle_odometry').value),
            self._odometry_callback,
            qos,
        )
        self.rates_pub = None
        if self.command_output_enabled:
            self.rates_pub = self.create_publisher(
                VehicleRatesSetpoint,
                str(self.get_parameter('topics.vehicle_rates_setpoint').value),
                qos,
            )
            self.get_logger().warning(
                'VISION-ONLY interception enabled; target-state inputs absent'
            )
        else:
            self.get_logger().warning(
                'Vision coordinator is debug-only; no PX4 rate publisher'
            )

    def _declare_visual_parameters(self) -> None:
        self.declare_parameter('confirmation_token', '')
        self.declare_parameter('feature_topic', '/interception/vision/raw_feature')
        self.declare_parameter('debug_topic', '/interception/control/debug')
        self.declare_parameter(
            'topics.vehicle_rates_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
        )
        self.declare_parameter(
            'topics.vehicle_odometry',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter('mass_kg', 2.0)
        self.declare_parameter('thrust_max_n', 26.9784)
        self.declare_parameter('hover_thrust_normalized', 0.75)
        self.declare_parameter('safe_los_angle_deg', 60.0)
        self.declare_parameter('los_rate_gain', 2.5)
        self.declare_parameter('image_center_rate_gain', 1.0)
        self.declare_parameter('attitude_rate_gain', 0.8)
        self.declare_parameter('approach_speed_gain', 1.2)
        self.declare_parameter('cruise_speed_m_s', 1.5)
        self.declare_parameter('terminal_speed_m_s', 2.2)
        self.declare_parameter('max_approach_acceleration_m_s2', 1.0)
        self.declare_parameter('max_approach_deceleration_m_s2', 1.0)
        self.declare_parameter('transverse_velocity_gain', 1.5)
        self.declare_parameter('max_transverse_acceleration_m_s2', 2.0)
        self.declare_parameter('vertical_velocity_gain', 2.0)
        self.declare_parameter('max_vertical_correction_m_s2', 2.0)
        self.declare_parameter('max_command_tilt_deg', 12.0)
        self.declare_parameter('image_area_px', 1228800.0)
        self.declare_parameter('terminal_area_ratio', 0.16)
        self.declare_parameter('terminal_max_center_error', 0.35)
        self.declare_parameter('full_speed_image_error', 0.08)
        self.declare_parameter('stop_approach_image_error', 0.30)
        self.declare_parameter('visual_search_yaw_rate_rad_s', 0.20)
        self.declare_parameter('visual_align_yaw_gain_rad_s', 1.2)
        self.declare_parameter('visual_align_vertical_gain_m_s', 0.4)
        self.declare_parameter('visual_align_center_error', 0.08)
        self.declare_parameter('visual_align_release_error', 0.14)
        self.declare_parameter('visual_align_settle_time_s', 0.5)
        self.declare_parameter('visual_search_loss_timeout_s', 0.3)
        self.declare_parameter('visual_align_max_vertical_offset_m', 0.6)
        self.declare_parameter('impact_loss_area_ratio', 0.04)
        self.declare_parameter('impact_loss_max_center_error', 0.60)
        self.declare_parameter('impact_loss_timeout_s', 0.30)
        self.declare_parameter('omega_limit_rad_s', 0.35)
        self.declare_parameter('yaw_rate_limit_rad_s', 0.25)
        self.declare_parameter('rate_slew_limit_rad_s2', 1.2)
        self.declare_parameter('rate_filter_time_constant_s', 0.08)
        self.declare_parameter('outer_loop_hz', 50.0)
        self.declare_parameter('interception_prestream_duration_s', 1.0)
        self.declare_parameter('mission_timeout_s', 60.0)
        self.declare_parameter('interception_telemetry_timeout_s', 0.5)
        self.declare_parameter('command_timeout_s', 0.1)
        self.declare_parameter('speed_limit_m_s', 4.5)
        self.declare_parameter('interception_tilt_limit_deg', 30.0)
        self.declare_parameter('minimum_barrier_margin', 0.02)
        self.declare_parameter('post_hit_coast_duration_s', 0.20)
        self.declare_parameter('stabilize_duration_s', 3.0)
        self.declare_parameter('stabilize_tilt_tolerance_deg', 8.0)
        self.declare_parameter('recovery_speed_tolerance_m_s', 0.35)
        self.declare_parameter('recovery_settle_time_s', 1.0)
        self.declare_parameter('recovery_timeout_s', 15.0)

    def _validate_activation(self) -> None:
        if not self.command_output_enabled:
            return
        token = str(self.get_parameter('confirmation_token').value)
        if token != VISION_CONTROL_CONFIRMATION_TOKEN:
            raise ValueError(
                'vision flight commands require ENABLE_VISION_ONLY_CONTROL'
            )

    def _feature_callback(self, message: VisionFeature) -> None:
        self.feature_received_ns = self.get_clock().now().nanoseconds
        self.last_feature = message
        self.target_detected = bool(message.valid)
        self.feature = message if message.valid else None
        if message.valid:
            self.last_valid_feature = message
            self.last_valid_feature_received_ns = self.feature_received_ns
        if not message.valid and self.terminal_started_s is None:
            self.control_reason = 'camera_target_not_detected'

    def _odometry_callback(self, message: VehicleOdometry) -> None:
        self.odometry = message
        self.odometry_received_ns = self.get_clock().now().nanoseconds

    def position_callback(self, message: VehicleLocalPosition) -> None:
        """Track only the interceptor's own safety and recovery state."""
        super().position_callback(message)
        velocity = np.asarray((message.vx, message.vy, message.vz), dtype=float)
        valid = bool(
            getattr(message, 'v_xy_valid', True)
            and message.v_z_valid
            and np.all(np.isfinite(velocity))
        )
        self.vehicle_speed_m_s = (
            float(np.linalg.norm(velocity)) if valid else 1e9
        )
        if valid:
            self.vehicle_velocity_ned = velocity
        heading = float(getattr(message, 'heading', math.nan))
        if math.isfinite(heading):
            self.vehicle_heading_ned_rad = heading

    def timer_callback(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        now_s = now_ns * 1e-9
        recovering = self.gate.phase in (
            InterceptionPhase.IMPACT_DETECTED,
            InterceptionPhase.EXIT_INTERCEPTION,
            InterceptionPhase.STABILIZE,
            InterceptionPhase.HOVER,
        )
        if recovering:
            self.visual_result = None
            self.px4_command = None
            self.control_reason = 'recovery_active'
        else:
            self._update_control(now_ns)
        if not self.command_output_enabled:
            self._publish_debug()
            return
        self._update_visual_terminal(now_s)
        self._run_enabled_step(now_s, now_ns)
        self._publish_debug()

    def _update_control(self, now_ns: int) -> None:
        if (
            self.last_outer_ns is not None
            and now_ns - self.last_outer_ns < self.outer_period_ns
        ):
            return
        self.last_outer_ns = now_ns
        feature = self.feature
        odometry = self.odometry
        if feature is None or odometry is None:
            self.visual_result = None
            self.px4_command = None
            return
        try:
            attitude = px4_quaternion_to_enu_flu_rotation(odometry.q)
            velocity_e = ned_to_enu(odometry.velocity)
            area_ratio = float(feature.area_px) / self.image_area_px
            result = compute_visual_ibvs(
                float(feature.x_norm),
                float(feature.y_norm),
                area_ratio,
                attitude,
                velocity_e,
                self.visual_config,
            )
            omega = self._bounded_rates(result.omega_d_b, now_ns)
            command = adapt_rate_thrust_command(
                omega,
                result.thrust_n,
                self.omega_limit_rad_s,
                self.mapping,
            )
        except (TypeError, ValueError) as error:
            self.visual_result = None
            self.px4_command = None
            self.control_reason = str(error)
            return
        self.visual_result = result
        self.px4_command = command
        self.command_computed_ns = now_ns
        self.control_reason = 'ok'

    def _bounded_rates(self, desired: np.ndarray, now_ns: int) -> np.ndarray:
        desired = np.asarray(desired, dtype=float).copy()
        desired[2] = float(np.clip(
            desired[2],
            -self.yaw_rate_limit_rad_s,
            self.yaw_rate_limit_rad_s,
        ))
        norm = float(np.linalg.norm(desired))
        if norm > self.omega_limit_rad_s:
            desired *= self.omega_limit_rad_s / norm
        previous_ns = self.last_rate_update_ns
        self.last_rate_update_ns = now_ns
        if previous_ns is None:
            self.filtered_omega_b = desired
            return desired.copy()
        elapsed_s = max(0.0, (now_ns - previous_ns) * 1e-9)
        smoothed = low_pass_rates(
            self.filtered_omega_b,
            desired,
            elapsed_s,
            self.rate_filter_time_constant_s,
        )
        delta = smoothed - self.filtered_omega_b
        maximum_change = self.rate_slew_limit_rad_s2 * elapsed_s
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > maximum_change and delta_norm > 1e-12:
            delta *= maximum_change / delta_norm
        self.filtered_omega_b = self.filtered_omega_b + delta
        return self.filtered_omega_b.copy()

    def _update_visual_terminal(self, now_s: float) -> None:
        if self.gate.phase != InterceptionPhase.ACTIVE:
            return
        result = self.visual_result
        feature = self.feature
        reason = ''
        if (
            self.terminal_started_s is None
            and result is not None
            and feature is not None
            and terminal_visual_ready(
                result.area_ratio,
                float(feature.x_norm),
                float(feature.y_norm),
                self.visual_config.terminal_area_ratio,
                self.terminal_center_error,
            )
        ):
            reason = 'target scale threshold'
        elif (
            self.terminal_started_s is None
            and feature is None
            and self.last_valid_feature is not None
            and self.last_valid_feature_received_ns is not None
        ):
            last = self.last_valid_feature
            age_s = max(
                0.0,
                now_s - self.last_valid_feature_received_ns * 1e-9,
            )
            if recent_close_target_lost(
                float(last.area_px) / self.image_area_px,
                float(last.x_norm),
                float(last.y_norm),
                age_s,
                self.impact_loss_area_ratio,
                self.impact_loss_center_error,
                self.impact_loss_timeout_s,
            ):
                reason = 'recent close target disappeared'
        if reason:
            self.terminal_started_s = now_s
            self.visual_interception_complete = True
            self._latch_coast_setpoint()
            self.get_logger().warning(
                f'Visual impact cue: exiting IBVS on {reason}'
            )

    def _run_enabled_step(self, now_s: float, now_ns: int) -> None:
        recovering = self.gate.phase in (
            InterceptionPhase.IMPACT_DETECTED,
            InterceptionPhase.EXIT_INTERCEPTION,
            InterceptionPhase.STABILIZE,
            InterceptionPhase.HOVER,
        )
        flight_actions = () if recovering else self.state_machine.step(now_s)
        self._log_phase_change()
        if not recovering and self.state_machine.phase != FlightPhase.HOLD:
            self._execute_actions(flight_actions)
            if (
                self.stop_on_complete
                and self.state_machine.phase == FlightPhase.COMPLETE
            ):
                self.finished = True
                self.timer.cancel()
            return

        if self.gate.phase == InterceptionPhase.WAIT_HOVER:
            self._update_visual_acquisition(now_s)

        actions = self.gate.step(
            now_s=now_s,
            hover_ready=bool(
                self.acquisition_command is not None
                and self.acquisition_command.ready
            ),
            landed=self.state_machine.landed,
            armed=self.state_machine.armed,
            offboard=self.state_machine.offboard,
            controller_valid=self.px4_command is not None,
            telemetry_age_s=self._telemetry_age_s(now_ns),
            command_age_s=self._command_age_s(now_ns),
            speed_m_s=self.vehicle_speed_m_s,
            tilt_rad=self.state_machine.tilt_rad,
            barrier_margin=self._barrier_margin(),
            interception_detected=self.visual_interception_complete,
            recovery_ready=self._stabilization_ready(now_s),
        )
        if self.gate.phase != self.last_gate_phase:
            self.get_logger().warning(
                f'Visual interception transition: '
                f'{self.last_gate_phase.value} -> {self.gate.phase.value}; '
                f'reason={self.gate.terminal_reason or "visual_control"}'
            )
            self.last_gate_phase = self.gate.phase
        if InterceptionAction.HOLD_POSITION in actions:
            self._publish_visual_acquisition_setpoint()
        elif InterceptionAction.STREAM_RATE_SETPOINT in actions:
            self._publish_rate_mode()
            if self.gate.phase == InterceptionPhase.PRESTREAM:
                self._publish_hover_rate_setpoint()
            else:
                self._publish_interception_setpoint()
        elif InterceptionAction.STREAM_EXIT_SETPOINT in actions:
            self._publish_exit_velocity_setpoint()
        elif InterceptionAction.STREAM_STABILIZE_SETPOINT in actions:
            self._publish_stabilize_velocity_setpoint(now_s)
        elif InterceptionAction.STREAM_HOVER_SETPOINT in actions:
            self._publish_recovery_position_setpoint()
        elif InterceptionAction.REQUEST_LAND in actions:
            self.state_machine.request_land(now_s)
            self._execute_actions(self.state_machine.step(now_s))

    def _publish_debug(self) -> None:
        message = ControlDebug()
        message.stamp = self.get_clock().now().to_msg()
        result = self.visual_result
        command = self.px4_command
        feature = self.feature or self.last_valid_feature or self.last_feature
        odometry = self.odometry
        message.target_detected = self.target_detected
        message.impact_detected = self.terminal_started_s is not None
        if (
            self.gate.phase == InterceptionPhase.WAIT_HOVER
            and self.state_machine.phase == FlightPhase.HOLD
            and self.acquisition_command is not None
        ):
            message.control_state = self.acquisition_command.phase.value
        else:
            message.control_state = self.gate.phase.value
        if feature is not None:
            message.image_error_norm = math.hypot(
                float(feature.x_norm),
                float(feature.y_norm),
            )
            message.target_area_ratio = (
                float(feature.area_px) / self.image_area_px
            )
        if odometry is not None:
            _assign_vector(
                message.attitude_rpy_rad,
                quaternion_wxyz_to_euler(odometry.q),
            )
            _assign_vector(
                message.angular_velocity_b,
                odometry.angular_velocity,
            )
            _assign_vector(message.velocity_ned, odometry.velocity)
        _assign_vector(message.commanded_body_rates, self.commanded_body_rates)
        _assign_vector(
            message.commanded_velocity_ned,
            self.commanded_velocity_ned,
        )
        _assign_vector(
            message.commanded_acceleration_ned,
            self.commanded_acceleration_ned,
        )
        message.commanded_thrust_normalized = (
            self.commanded_thrust_normalized
        )
        if result is None or command is None:
            message.valid = False
            message.reason = self.control_reason
            self.debug_pub.publish(message)
            return
        message.z1 = result.z1
        if self.feature is not None:
            _assign_vector(
                message.z2,
                (
                    self.feature.x_norm,
                    self.feature.y_norm,
                    result.area_ratio,
                ),
            )
        _assign_vector(message.acceleration_d, result.acceleration_d_e)
        message.attitude_d = tuple(result.attitude_d_b_to_e.reshape(9))
        _assign_vector(message.omega1_b, result.omega_los_b)
        _assign_vector(message.omega2_b, result.omega_attitude_b)
        _assign_vector(message.omega_d_b, self.filtered_omega_b)
        message.thrust_n = result.thrust_n
        message.thrust_normalized = -command.thrust_body[2]
        message.thrust_saturated = result.thrust_saturated
        message.thrust_mapping_saturated = command.thrust_saturated
        message.rate_saturated = bool(
            np.linalg.norm(result.omega_d_b) > self.omega_limit_rad_s
        )
        message.valid = True
        message.reason = 'ok_visual_features_only'
        self.debug_pub.publish(message)

    def _publish_rate_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.body_rate = True
        self.offboard_pub.publish(message)

    def _update_visual_acquisition(self, now_s: float) -> None:
        """Search and center using pixels while PX4 holds local position."""
        heading = self.vehicle_heading_ned_rad
        if heading is None:
            return
        feature = self.feature
        command = self.acquisition.step(
            now_s=now_s,
            current_yaw_rad=heading,
            target_detected=feature is not None,
            x_norm=float(feature.x_norm) if feature is not None else 0.0,
            y_norm=float(feature.y_norm) if feature is not None else 0.0,
        )
        self.acquisition_command = command
        if command.phase != self.last_acquisition_phase:
            self.get_logger().warning(
                f'Visual acquisition transition: '
                f'{self.last_acquisition_phase.value} -> '
                f'{command.phase.value}'
            )
            self.last_acquisition_phase = command.phase

    def _publish_visual_acquisition_setpoint(self) -> None:
        """Hold position while search/alignment changes only yaw and height."""
        command = self.acquisition_command
        target_z = self.state_machine.target_z_ned_m
        home_x = self.state_machine.home_x
        home_y = self.state_machine.home_y
        if (
            command is None
            or target_z is None
            or home_x is None
            or home_y is None
        ):
            return
        mode = OffboardControlMode()
        mode.timestamp = self._timestamp_us()
        mode.position = True
        self.offboard_pub.publish(mode)
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = [
            home_x,
            home_y,
            target_z + command.vertical_offset_m,
        ]
        setpoint.velocity = [math.nan, math.nan, math.nan]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = command.yaw_setpoint_rad
        setpoint.yawspeed = math.nan
        self.trajectory_pub.publish(setpoint)
        self.commanded_body_rates.fill(math.nan)
        self.commanded_velocity_ned.fill(0.0)
        self.commanded_acceleration_ned.fill(0.0)
        self.commanded_thrust_normalized = math.nan

    def _publish_hover_rate_setpoint(self) -> None:
        command = adapt_rate_thrust_command(
            (0.0, 0.0, 0.0),
            self.visual_config.mass_kg * self.visual_config.gravity_m_s2,
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
        self.commanded_body_rates = np.asarray(
            command.rates_frd_rad_s,
            dtype=float,
        )
        self.commanded_velocity_ned.fill(math.nan)
        self.commanded_acceleration_ned.fill(math.nan)
        self.commanded_thrust_normalized = -float(command.thrust_body[2])

    def _latch_coast_setpoint(self) -> None:
        if self.coast_velocity_ned is not None:
            return
        velocity = self.vehicle_velocity_ned
        if velocity is None or not np.all(np.isfinite(velocity)):
            velocity = np.zeros(3)
        self.coast_velocity_ned = velocity.copy()
        self.coast_yaw_ned_rad = (
            self.vehicle_heading_ned_rad
            if self.vehicle_heading_ned_rad is not None
            else self.target_yaw_rad
        )

    def _publish_exit_velocity_setpoint(self) -> None:
        """Keep the setpoint continuous while clearing the airframe."""
        self._latch_coast_setpoint()
        velocity = self.coast_velocity_ned
        acceleration = np.zeros(3)
        self._publish_velocity_setpoint(velocity, acceleration)

    def _publish_stabilize_velocity_setpoint(self, now_s: float) -> None:
        """Smoothly remove interception velocity without reversing it."""
        self._latch_coast_setpoint()
        if self.stabilize_started_s is None:
            self.stabilize_started_s = now_s
        command = smooth_stop_command(
            self.coast_velocity_ned,
            max(0.0, now_s - self.stabilize_started_s),
            self.stabilize_duration_s,
        )
        self._publish_velocity_setpoint(
            command.velocity_ned,
            command.acceleration_ned,
        )

    def _publish_velocity_setpoint(
        self,
        velocity_ned: np.ndarray,
        acceleration_ned: np.ndarray,
    ) -> None:
        """Publish one finite, auditable velocity-mode transition command."""
        mode = OffboardControlMode()
        mode.timestamp = self._timestamp_us()
        mode.velocity = True
        self.offboard_pub.publish(mode)
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = [math.nan, math.nan, math.nan]
        setpoint.velocity = list(velocity_ned)
        setpoint.acceleration = list(acceleration_ned)
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = self.coast_yaw_ned_rad
        setpoint.yawspeed = math.nan
        self.trajectory_pub.publish(setpoint)
        self.commanded_body_rates.fill(math.nan)
        self.commanded_velocity_ned = np.asarray(
            velocity_ned,
            dtype=float,
        ).copy()
        self.commanded_acceleration_ned = np.asarray(
            acceleration_ned,
            dtype=float,
        ).copy()
        self.commanded_thrust_normalized = math.nan

    def _latch_recovery_setpoint(self) -> None:
        if self.recovery_target_z_ned is not None:
            return
        configured_z = self.state_machine.target_z_ned_m
        if configured_z is None:
            return
        self.recovery_hold_x_ned = self.state_machine.x
        self.recovery_hold_y_ned = self.state_machine.y
        self.recovery_target_z_ned = min(
            self.state_machine.z,
            configured_z,
        )

    def _publish_recovery_position_setpoint(self) -> None:
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
        self.commanded_body_rates.fill(math.nan)
        self.commanded_velocity_ned.fill(0.0)
        self.commanded_acceleration_ned.fill(0.0)
        self.commanded_thrust_normalized = math.nan

    def _stabilization_ready(self, now_s: float) -> bool:
        if self.stabilize_started_s is None:
            return False
        return (
            now_s - self.stabilize_started_s >= self.stabilize_duration_s
            and self.state_machine.tilt_rad
            <= self.stabilize_tilt_tolerance_rad
            and self.vehicle_speed_m_s
            <= self._float_parameter('recovery_speed_tolerance_m_s')
        )

    def _telemetry_age_s(self, now_ns: int) -> float:
        timestamps = (self.feature_received_ns, self.odometry_received_ns)
        if None in timestamps:
            return 1e9
        return max(now_ns - timestamp for timestamp in timestamps) * 1e-9

    def _command_age_s(self, now_ns: int) -> float:
        if self.command_computed_ns is None:
            return 1e9
        return (now_ns - self.command_computed_ns) * 1e-9

    def _barrier_margin(self) -> float:
        if self.visual_result is None:
            # During the bounded terminal commit, the most recent command and
            # its already-validated visual barrier remain authoritative.
            return self.k_b if self.terminal_started_s is not None else -1e9
        return self.visual_result.barrier_margin


def main(args=None) -> None:
    """Run the integrated camera-feature-only coordinator."""
    import rclpy

    rclpy.init(args=args)
    node = VisionInterceptionCoordinator()
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
