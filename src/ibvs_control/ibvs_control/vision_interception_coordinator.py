"""Observer-fed implementation of the paper interception controller."""

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

from interception_interfaces.msg import (
    ControlDebug,
    ObserverBootstrap,
    ObserverState,
    VisionFeature,
)
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
from std_msgs.msg import Bool, Empty

from ibvs_control.frames import (
    ned_to_enu,
    px4_quaternion_to_enu_flu_rotation,
    quaternion_wxyz_to_euler,
)
from ibvs_control.bearing_range_estimator import (
    BearingRangeConfig,
    BearingRangeEstimator,
    BearingSample,
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
from ibvs_control.so3_controller import (
    InnerLoopResult,
    OuterLoopConfig,
    OuterLoopResult,
    compute_drag_force_e,
    compute_inner_loop,
    compute_outer_loop,
    image_los_in_earth,
)
from ibvs_control.takeoff_state_machine import FlightPhase
from ibvs_control.thrust_mapping import ThrustMappingConfig
from ibvs_control.visual_acquisition import (
    VisualAcquisition,
    VisualAcquisitionCommand,
    VisualAcquisitionConfig,
)


VISION_CONTROL_CONFIRMATION_TOKEN = 'ENABLE_VISION_ONLY_CONTROL'


@dataclass(frozen=True)
class _CameraPose:
    """One onboard pose tagged with its ROS clock receipt time."""

    stamp_ns: int
    vehicle_position_e: np.ndarray
    camera_position_e: np.ndarray
    camera_to_earth: np.ndarray


def _assign_vector(message, values) -> None:
    """Copy a three-element iterable into a ROS Vector3 message."""
    message.x, message.y, message.z = (float(value) for value in values)


def _clip(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar without introducing another control dependency."""
    return min(max(value, lower), upper)


class VisionInterceptionCoordinator(OffboardTakeoff):
    """
    Take off and intercept using the paper observer and Section III law.

    Raw image features are used only for target acquisition and diagnostics.
    Once interception is active, target bearing and relative state come only
    from the delayed observer defined by the paper.
    """

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
        self.paper_config = OuterLoopConfig(
            k1=self._float_parameter('paper_k1'),
            k2=self._float_parameter('paper_k2'),
            position_gain=self._float_parameter('position_gain'),
            k_b=self.k_b,
            mass_kg=self._float_parameter('mass_kg'),
            thrust_max_n=self._float_parameter('thrust_max_n'),
        )
        self.paper_config.validate()
        drag_coefficients = np.asarray(
            self.get_parameter('drag_coefficients_b_kg_s').value,
            dtype=float,
        )
        if (
            drag_coefficients.shape != (3,)
            or not np.all(np.isfinite(drag_coefficients))
            or np.any(drag_coefficients < 0.0)
        ):
            raise ValueError(
                'drag_coefficients_b_kg_s must contain three finite '
                'nonnegative values'
            )
        self.drag_coefficients_b = tuple(
            float(value) for value in drag_coefficients
        )
        camera_rotation = np.asarray(
            self.get_parameter('camera_to_body_rotation').value,
            dtype=float,
        )
        if camera_rotation.shape != (9,):
            raise ValueError('camera_to_body_rotation must contain 9 values')
        self.camera_to_body_rotation = camera_rotation.reshape((3, 3))
        self.camera_position_b_m = np.asarray(
            self.get_parameter('camera_position_b_m').value, dtype=float
        )
        if (
            self.camera_position_b_m.shape != (3,)
            or not np.all(np.isfinite(self.camera_position_b_m))
        ):
            raise ValueError(
                'camera_position_b_m must contain 3 finite values'
            )
        self.bootstrap_pose_max_error_ns = int(
            self._float_parameter('range_pose_max_time_error_s') * 1e9
        )
        self.bootstrap_pose_history_ns = int(
            self._float_parameter('range_pose_history_s') * 1e9
        )
        self.bootstrap_position_std_floor_m = self._float_parameter(
            'range_position_std_floor_m'
        )
        self.bootstrap_velocity_std_floor_m_s = self._float_parameter(
            'range_velocity_std_floor_m_s'
        )
        self.bootstrap_max_age_s = self._float_parameter(
            'range_bootstrap_max_age_s'
        )
        self.min_observable_duration_s = self._float_parameter(
            'range_min_observable_duration_s'
        )
        self.bootstrap_max_relative_std = self._float_parameter(
            'range_max_relative_std'
        )
        self.range_probe_amplitude_m = self._float_parameter(
            'range_probe_vertical_amplitude_m'
        )
        self.range_probe_period_s = self._float_parameter(
            'range_probe_vertical_period_s'
        )
        if min(
            self.bootstrap_pose_max_error_ns,
            self.bootstrap_pose_history_ns,
            self.bootstrap_position_std_floor_m,
            self.bootstrap_velocity_std_floor_m_s,
            self.bootstrap_max_age_s,
            self.min_observable_duration_s,
            self.bootstrap_max_relative_std,
            self.range_probe_amplitude_m,
            self.range_probe_period_s,
        ) <= 0:
            raise ValueError('range bootstrap limits must be positive')
        self.range_estimator = BearingRangeEstimator(BearingRangeConfig(
            model=str(self.get_parameter('range_target_model').value),
            image_noise_std=self._float_parameter('range_image_noise_std'),
            min_samples=self._int_parameter('range_min_samples'),
            max_samples=self._int_parameter('range_max_samples'),
            max_age_s=self._float_parameter('range_window_s'),
            min_parallax_deg=self._float_parameter('range_min_parallax_deg'),
            min_transverse_baseline_m=self._float_parameter(
                'range_min_transverse_baseline_m'
            ),
            min_time_span_s=self._float_parameter('range_min_time_span_s'),
            max_condition_number=self._float_parameter(
                'range_max_condition_number'
            ),
            max_reprojection_rms=self._float_parameter(
                'range_max_reprojection_rms'
            ),
            max_range_std_m=self._float_parameter('range_max_std_m'),
            max_velocity_std_m_s=self._float_parameter(
                'range_max_velocity_std_m_s'
            ),
            outlier_reprojection_threshold=self._float_parameter(
                'range_outlier_reprojection_threshold'
            ),
            max_outlier_fraction=self._float_parameter(
                'range_max_outlier_fraction'
            ),
        ))
        self.bootstrap_poses: deque[_CameraPose] = deque()
        self.range_estimate = None
        self.last_range_fit_ns: Optional[int] = None
        self.first_accepted_range_ns: Optional[int] = None
        self.bootstrap_sent = False
        self.last_range_reason = ''
        self.range_block_reason = 'waiting_for_time_aligned_bearings'
        self.range_probe_started_s: Optional[float] = None
        # The configured pitch is positive from body-forward toward body-up.
        # The camera-centre trial uses zero so the controlled LOS and the
        # camera optical axis are the same physical impact line.
        designed_pitch = math.radians(
            self._float_parameter('designed_los_pitch_deg')
        )
        if not -math.pi / 2.0 < designed_pitch < math.pi / 2.0:
            raise ValueError(
                'designed_los_pitch_deg must be within (-90, 90)'
            )
        self.designed_los_b = np.array(
            (math.cos(designed_pitch), 0.0, math.sin(designed_pitch))
        )
        designed_los_c = (
            self.camera_to_body_rotation.T @ self.designed_los_b
        )
        if designed_los_c[2] <= 1e-6:
            raise ValueError('designed LOS must remain in front of the camera')
        self.designed_image_xy = (
            designed_los_c[:2] / designed_los_c[2]
        )
        minimum_barrier_margin = self._float_parameter(
            'minimum_barrier_margin'
        )
        start_reserve = self._float_parameter(
            'visual_start_barrier_margin_reserve'
        )
        release_reserve = self._float_parameter(
            'visual_release_barrier_margin_reserve'
        )
        if not 0.0 < release_reserve < start_reserve:
            raise ValueError(
                'visual barrier reserves must satisfy '
                '0 < release < start'
            )
        self.acquisition = VisualAcquisition(
            VisualAcquisitionConfig(
                search_yaw_rate_rad_s=self._float_parameter(
                    'visual_search_yaw_rate_rad_s'
                ),
                search_vertical_amplitude_m=self._float_parameter(
                    'visual_search_vertical_amplitude_m'
                ),
                search_vertical_period_s=self._float_parameter(
                    'visual_search_vertical_period_s'
                ),
                align_yaw_gain_rad_s=self._float_parameter(
                    'visual_align_yaw_gain_rad_s'
                ),
                align_vertical_gain_m_s=self._float_parameter(
                    'visual_align_vertical_gain_m_s'
                ),
                k_b=self.k_b,
                desired_image_xy=tuple(self.designed_image_xy),
                start_barrier_margin=(
                    minimum_barrier_margin + start_reserve
                ),
                release_barrier_margin=(
                    minimum_barrier_margin + release_reserve
                ),
                settle_time_s=self._float_parameter(
                    'visual_align_settle_time_s'
                ),
                target_loss_timeout_s=self._float_parameter(
                    'visual_search_loss_timeout_s'
                ),
            )
        )
        self.mapping = ThrustMappingConfig(
            mass_kg=self.paper_config.mass_kg,
            hover_thrust_normalized=self._float_parameter(
                'hover_thrust_normalized'
            ),
            maximum_thrust_n=self.paper_config.thrust_max_n,
            thrust_curve_exponent=self._float_parameter(
                'thrust_curve_exponent'
            ),
            actuator_minimum_fraction=self._float_parameter(
                'actuator_minimum_fraction'
            ),
        )
        self.mapping.validate()
        self.image_area_px = self._float_parameter('image_area_px')
        self.visual_feature_timeout_s = self._float_parameter(
            'visual_feature_timeout_s'
        )
        self.stabilize_duration_s = self._float_parameter(
            'stabilize_duration_s'
        )
        self.stabilize_tilt_tolerance_rad = math.radians(
            self._float_parameter('stabilize_tilt_tolerance_deg')
        )
        if self.image_area_px <= 0.0:
            raise ValueError('image_area_px must be positive')
        if self.visual_feature_timeout_s <= 0.0:
            raise ValueError('visual_feature_timeout_s must be positive')
        if self.stabilize_duration_s <= 0.0:
            raise ValueError('stabilize_duration_s must be positive')
        if not 0.0 < self.stabilize_tilt_tolerance_rad < math.pi / 2.0:
            raise ValueError('stabilize tilt tolerance must be within (0, 90)')

        self.omega_limit_rad_s = self._float_parameter(
            'omega_limit_rad_s'
        )
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
        self.target_detected = False
        self.contact_confirmed = False
        self.observer_state: Optional[ObserverState] = None
        self.odometry: Optional[VehicleOdometry] = None
        self.feature_received_ns: Optional[int] = None
        self.observer_received_ns: Optional[int] = None
        self.odometry_received_ns: Optional[int] = None
        self.visual_result: Optional[OuterLoopResult] = None
        self.inner_result: Optional[InnerLoopResult] = None
        self.px4_command: Optional[Px4RateThrustCommand] = None
        self.command_computed_ns: Optional[int] = None
        self.control_reason = 'waiting_for_camera_feature'
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
        self.observer_reset_pub = self.create_publisher(
            Empty,
            str(self.get_parameter('observer_reset_topic').value),
            10,
        )
        self.observer_bootstrap_pub = self.create_publisher(
            ObserverBootstrap,
            str(self.get_parameter('observer_bootstrap_topic').value),
            10,
        )
        self.create_subscription(
            VisionFeature,
            str(self.get_parameter('feature_topic').value),
            self._feature_callback,
            qos,
        )
        self.create_subscription(
            ObserverState,
            str(self.get_parameter('observer_topic').value),
            self._observer_callback,
            qos,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter('topics.vehicle_odometry').value),
            self._odometry_callback,
            qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('target_contact_topic').value),
            self._contact_callback,
            10,
        )
        self.rates_pub = None
        if self.command_output_enabled:
            self.rates_pub = self.create_publisher(
                VehicleRatesSetpoint,
                str(self.get_parameter('topics.vehicle_rates_setpoint').value),
                qos,
            )
            self.get_logger().warning(
                'Paper observer/controller enabled; target truth inputs absent'
            )
        else:
            self.get_logger().warning(
                'Vision coordinator is debug-only; no PX4 rate publisher'
            )

    def _declare_visual_parameters(self) -> None:
        self.declare_parameter('confirmation_token', '')
        self.declare_parameter(
            'feature_topic',
            '/interception/vision/raw_feature',
        )
        self.declare_parameter(
            'observer_topic',
            '/interception/observer/state',
        )
        self.declare_parameter(
            'observer_reset_topic',
            '/interception/observer/reset',
        )
        self.declare_parameter(
            'observer_bootstrap_topic',
            '/interception/observer/bootstrap',
        )
        self.declare_parameter('debug_topic', '/interception/control/debug')
        self.declare_parameter(
            'target_contact_topic',
            '/interception/target/green_confirmed',
        )
        self.declare_parameter(
            'topics.vehicle_rates_setpoint',
            '/fmu/in/vehicle_rates_setpoint',
        )
        self.declare_parameter(
            'topics.vehicle_odometry',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter('mass_kg', 2.1143076923)
        self.declare_parameter('thrust_max_n', 58.8399)
        # Equivalent linearized x500 rotor drag for the paper T/W=3 baseline.
        self.declare_parameter(
            'drag_coefficients_b_kg_s',
            [0.1862366, 0.1862366, 0.0],
        )
        self.declare_parameter('hover_thrust_normalized', 0.5219057941)
        self.declare_parameter('thrust_curve_exponent', 2.0)
        self.declare_parameter('actuator_minimum_fraction', 0.15)
        self.declare_parameter('safe_los_angle_deg', 60.0)
        self.declare_parameter('paper_k1', 0.05)
        self.declare_parameter('paper_k2', 3.0)
        self.declare_parameter('position_gain', 1.0)
        self.declare_parameter('designed_los_pitch_deg', 0.0)
        self.declare_parameter(
            'camera_to_body_rotation',
            [
                0.0, 0.0, 1.0,
                -1.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
            ],
        )
        self.declare_parameter('camera_position_b_m', [0.12, 0.0, 0.0])
        self.declare_parameter('range_target_model', 'static')
        self.declare_parameter('range_image_noise_std', 0.015)
        self.declare_parameter('range_min_samples', 20)
        self.declare_parameter('range_max_samples', 500)
        self.declare_parameter('range_window_s', 30.0)
        self.declare_parameter('range_min_parallax_deg', 1.0)
        self.declare_parameter('range_min_transverse_baseline_m', 2.0)
        self.declare_parameter('range_min_time_span_s', 2.0)
        self.declare_parameter('range_max_condition_number', 100.0)
        self.declare_parameter('range_max_reprojection_rms', 0.03)
        self.declare_parameter('range_max_std_m', 2.0)
        self.declare_parameter('range_max_velocity_std_m_s', 2.0)
        self.declare_parameter('range_outlier_reprojection_threshold', 0.06)
        self.declare_parameter('range_max_outlier_fraction', 0.2)
        self.declare_parameter('range_pose_max_time_error_s', 0.03)
        self.declare_parameter('range_pose_history_s', 35.0)
        self.declare_parameter('range_position_std_floor_m', 1.0)
        self.declare_parameter('range_velocity_std_floor_m_s', 5.0)
        self.declare_parameter('range_bootstrap_max_age_s', 0.5)
        self.declare_parameter('range_min_observable_duration_s', 2.0)
        self.declare_parameter('range_max_relative_std', 0.2)
        self.declare_parameter('range_probe_vertical_amplitude_m', 1.5)
        self.declare_parameter('range_probe_vertical_period_s', 8.0)
        self.declare_parameter('image_area_px', 1228800.0)
        self.declare_parameter('visual_feature_timeout_s', 0.20)
        self.declare_parameter('visual_search_yaw_rate_rad_s', 0.35)
        self.declare_parameter('visual_search_vertical_amplitude_m', 3.0)
        self.declare_parameter('visual_search_vertical_period_s', 20.0)
        self.declare_parameter('visual_align_yaw_gain_rad_s', 1.2)
        self.declare_parameter('visual_align_vertical_gain_m_s', 1.0)
        self.declare_parameter('visual_start_barrier_margin_reserve', 0.02)
        self.declare_parameter('visual_release_barrier_margin_reserve', 0.005)
        self.declare_parameter('visual_align_settle_time_s', 0.5)
        self.declare_parameter('visual_search_loss_timeout_s', 0.3)
        self.declare_parameter('omega_limit_rad_s', 0.20)
        self.declare_parameter('outer_loop_hz', 50.0)
        self.declare_parameter('interception_prestream_duration_s', 1.0)
        self.declare_parameter('mission_timeout_s', 60.0)
        self.declare_parameter('interception_telemetry_timeout_s', 0.5)
        self.declare_parameter('command_timeout_s', 0.2)
        # 拦截阶段最大飞行速度，单位m/s；超过后由安全门中止任务。
        self.declare_parameter('speed_limit_m_s', 21.0)
        # 拦截阶段最大倾角，单位deg；超过后由安全门中止任务。
        self.declare_parameter('interception_tilt_limit_deg', 55.0)
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
        if not message.valid and self.terminal_started_s is None:
            self.control_reason = 'camera_target_not_detected'
        if (
            message.valid
            and not self.bootstrap_sent
            and self.state_machine.phase in (FlightPhase.TAKEOFF,
                                             FlightPhase.HOLD)
        ):
            self._record_range_bearing(message)

    def _observer_callback(self, message: ObserverState) -> None:
        """Accept the sole state input used by the paper controller."""
        self.observer_received_ns = self.get_clock().now().nanoseconds
        if message.initialized and message.valid:
            self.observer_state = message
        else:
            self.observer_state = None
            self.control_reason = message.reason or 'observer_invalid'

    def _odometry_callback(self, message: VehicleOdometry) -> None:
        self.odometry = message
        self.odometry_received_ns = self.get_clock().now().nanoseconds
        try:
            vehicle_position_e = np.asarray(
                ned_to_enu(message.position), dtype=float
            )
            rotation_b_to_e = px4_quaternion_to_enu_flu_rotation(
                message.q
            )
            camera_position_e = (
                vehicle_position_e
                + rotation_b_to_e @ self.camera_position_b_m
            )
            pose = _CameraPose(
                stamp_ns=self.odometry_received_ns,
                vehicle_position_e=vehicle_position_e,
                camera_position_e=camera_position_e,
                camera_to_earth=(
                    rotation_b_to_e @ self.camera_to_body_rotation
                ),
            )
            if (
                self.bootstrap_poses
                and pose.stamp_ns <= self.bootstrap_poses[-1].stamp_ns
            ):
                self.bootstrap_poses.clear()
                self.range_estimator.clear()
                self.range_estimate = None
                self.last_range_fit_ns = None
                self.first_accepted_range_ns = None
            self.bootstrap_poses.append(pose)
            cutoff_ns = pose.stamp_ns - self.bootstrap_pose_history_ns
            while (
                self.bootstrap_poses
                and self.bootstrap_poses[0].stamp_ns < cutoff_ns
            ):
                self.bootstrap_poses.popleft()
        except (TypeError, ValueError, np.linalg.LinAlgError):
            # The normal telemetry freshness checks continue to guard flight.
            return

    def _matching_bootstrap_pose(
        self, stamp_ns: int
    ) -> Optional[_CameraPose]:
        """Return an onboard pose close to an image capture instant."""
        if not self.bootstrap_poses:
            return None
        pose = min(
            self.bootstrap_poses,
            key=lambda item: abs(item.stamp_ns - stamp_ns),
        )
        if abs(pose.stamp_ns - stamp_ns) > self.bootstrap_pose_max_error_ns:
            return None
        return pose

    def _record_range_bearing(self, message: VisionFeature) -> None:
        """Update the pre-interception range estimate from captured rays."""
        capture_ns = (
            int(message.capture_stamp.sec) * 1_000_000_000
            + int(message.capture_stamp.nanosec)
        )
        now_ns = self.get_clock().now().nanoseconds
        if (
            capture_ns <= 0
            or capture_ns > now_ns + self.bootstrap_pose_max_error_ns
            or now_ns - capture_ns > self.bootstrap_pose_history_ns
        ):
            return
        pose = self._matching_bootstrap_pose(capture_ns)
        if pose is None:
            return
        try:
            added = self.range_estimator.add(BearingSample(
                timestamp_s=capture_ns * 1e-9,
                image_xy=(float(message.x_norm), float(message.y_norm)),
                camera_position_enu_m=tuple(pose.camera_position_e),
                camera_to_enu_rotation=pose.camera_to_earth,
            ))
            if not added:
                return
            if (
                self.last_range_fit_ns is None
                or capture_ns - self.last_range_fit_ns >= 200_000_000
            ):
                self.range_estimate = self.range_estimator.estimate()
                self.last_range_fit_ns = capture_ns
                if self.range_estimate.accepted:
                    if self.first_accepted_range_ns is None:
                        self.first_accepted_range_ns = capture_ns
                else:
                    self.first_accepted_range_ns = None
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            reason = f'range_estimation_error: {error}'
            if reason != self.last_range_reason:
                self.get_logger().warning(reason)
                self.last_range_reason = reason

    def _range_bootstrap_message(
        self, now_ns: int
    ) -> Optional[ObserverBootstrap]:
        """Convert an accepted target fit into observer relative state."""
        estimate = self.range_estimate
        if (
            estimate is None
            or not estimate.accepted
            or not self.bootstrap_poses
            or estimate.reference_time_s is None
        ):
            self.range_block_reason = (
                estimate.reason if estimate is not None
                else 'waiting_for_time_aligned_bearings'
            )
            return None
        if (
            self.first_accepted_range_ns is None
            or now_ns - self.first_accepted_range_ns
            < self.min_observable_duration_s * 1e9
        ):
            self.range_block_reason = 'observability_settling'
            return None
        delta_s = now_ns * 1e-9 - estimate.reference_time_s
        if not 0.0 <= delta_s <= self.bootstrap_max_age_s:
            self.range_block_reason = 'range_estimate_stale'
            return None
        pose = self.bootstrap_poses[-1]
        if now_ns - pose.stamp_ns > self.bootstrap_pose_max_error_ns:
            self.range_block_reason = 'vehicle_pose_stale'
            return None
        velocity_ned = self.vehicle_velocity_ned
        if velocity_ned is None and self.odometry is not None:
            velocity_ned = np.asarray(self.odometry.velocity, dtype=float)
        if velocity_ned is None or not np.all(np.isfinite(velocity_ned)):
            self.range_block_reason = 'vehicle_velocity_unavailable'
            return None
        vehicle_velocity_e = np.asarray(ned_to_enu(velocity_ned))
        target_position_e = estimate.target_position_enu_m.copy()
        target_velocity_e = (
            estimate.target_velocity_enu_m_s.copy()
            if estimate.target_velocity_enu_m_s is not None
            else np.zeros(3)
        )
        covariance = np.asarray(estimate.covariance_enu, dtype=float)
        relative_covariance = np.zeros((6, 6))
        if covariance.shape == (6, 6):
            transition = np.eye(6)
            transition[:3, 3:] = np.eye(3) * delta_s
            relative_covariance = (
                transition @ covariance @ transition.T
            )
            target_position_e += target_velocity_e * delta_s
        elif covariance.shape == (3, 3):
            relative_covariance[:3, :3] = covariance
        else:
            self.range_block_reason = 'invalid_estimate_covariance'
            return None
        relative_covariance[:3, :3] += (
            self.bootstrap_position_std_floor_m ** 2 * np.eye(3)
        )
        relative_covariance[3:, 3:] += (
            self.bootstrap_velocity_std_floor_m_s ** 2 * np.eye(3)
        )
        relative_covariance = (
            relative_covariance + relative_covariance.T
        ) / 2.0
        p_r_e = pose.vehicle_position_e - target_position_e
        v_r_e = vehicle_velocity_e - target_velocity_e
        target_c = pose.camera_to_earth.T @ (
            target_position_e - pose.camera_position_e
        )
        if target_c[2] <= self.range_estimator.config.min_depth_m:
            self.range_block_reason = 'target_behind_camera'
            return None
        range_m = float(np.linalg.norm(p_r_e))
        if range_m <= 0.0:
            self.range_block_reason = 'invalid_estimated_range'
            return None
        radial = p_r_e / range_m
        range_std_m = math.sqrt(max(
            0.0,
            float(radial @ relative_covariance[:3, :3] @ radial),
        ))
        if (
            range_std_m > self.range_estimator.config.max_range_std_m
            or range_std_m / range_m > self.bootstrap_max_relative_std
        ):
            self.range_block_reason = 'uncertain_relative_range'
            return None
        self.range_block_reason = 'ok'
        bootstrap = ObserverBootstrap()
        bootstrap.stamp = self.get_clock().now().to_msg()
        _assign_vector(bootstrap.p_r, p_r_e)
        _assign_vector(bootstrap.v_r, v_r_e)
        bootstrap.relative_covariance = tuple(
            relative_covariance.reshape(-1)
        )
        bootstrap.sample_count = estimate.quality.sample_count
        bootstrap.range_m = range_m
        bootstrap.range_std_m = range_std_m
        return bootstrap

    def _contact_callback(self, message: Bool) -> None:
        if message.data:
            self.contact_confirmed = True
            self.get_logger().warning(
                'Gazebo target contact confirmed; committing visual impact'
            )

    def position_callback(self, message: VehicleLocalPosition) -> None:
        """Track only the interceptor's own safety and recovery state."""
        super().position_callback(message)
        velocity = np.asarray(
            (message.vx, message.vy, message.vz),
            dtype=float,
        )
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
            self.inner_result = None
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
        observer = self.observer_state
        paper_result: Optional[OuterLoopResult] = None
        paper_inner: Optional[InnerLoopResult] = None

        # Algorithm 1 takes target relative state from the observer and the
        # interceptor's current R_b^e from onboard attitude telemetry.
        if observer is not None and self.odometry is not None:
            try:
                vehicle_attitude = px4_quaternion_to_enu_flu_rotation(
                    self.odometry.q
                )
                p_r_e, v_r_e, image_xy = self._controller_state(observer)
                # VehicleLocalPosition is explicitly NED and is preferred
                # here.  The odometry fallback preserves the controller's
                # existing NED convention for startup/debug-only cases.
                velocity_ned = self.vehicle_velocity_ned
                if velocity_ned is None:
                    velocity_ned = np.asarray(
                        self.odometry.velocity,
                        dtype=float,
                    )
                velocity_e = np.asarray(
                    ned_to_enu(velocity_ned),
                    dtype=float,
                )
                drag_force_e = compute_drag_force_e(
                    velocity_e,
                    vehicle_attitude,
                    self.drag_coefficients_b,
                )
                outer_due = bool(
                    self.visual_result is None
                    or self.last_outer_ns is None
                    or now_ns - self.last_outer_ns >= self.outer_period_ns
                )
                if outer_due:
                    los_e = image_los_in_earth(
                        image_xy,
                        vehicle_attitude,
                        self.camera_to_body_rotation,
                    )
                    designed_los_e = vehicle_attitude @ self.designed_los_b
                    paper_result = compute_outer_loop(
                        p_r_e,
                        v_r_e,
                        los_e,
                        designed_los_e,
                        vehicle_attitude,
                        self.paper_config,
                        drag_force_e=drag_force_e,
                    )
                    self.last_outer_ns = now_ns
                else:
                    paper_result = self.visual_result
                if paper_result is not None:
                    paper_inner = compute_inner_loop(
                        paper_result,
                        vehicle_attitude,
                        self.paper_config,
                        self.omega_limit_rad_s,
                        drag_force_e=drag_force_e,
                    )
            except (TypeError, ValueError) as error:
                self.control_reason = f'paper_observer_invalid: {error}'

        if paper_inner is None:
            # Retain the most recent valid command for the bounded freshness
            # window enforced by command_timeout_s. Clearing it here made a
            # single invalid DKF sample abort immediately and bypassed that
            # gate, including the minimum-depth singularity just before
            # physical target contact.
            if self.px4_command is not None:
                if not self.control_reason.startswith(
                    'holding_last_valid_command:'
                ):
                    self.control_reason = (
                        f'holding_last_valid_command: {self.control_reason}'
                    )
            return

        try:
            command = adapt_rate_thrust_command(
                paper_inner.omega_d_b,
                paper_inner.thrust_n,
                self.omega_limit_rad_s,
                self.mapping,
            )
        except (TypeError, ValueError) as error:
            self.control_reason = (
                f'holding_last_valid_command: {error}'
                if self.px4_command is not None else str(error)
            )
            return
        self.visual_result = paper_result
        self.inner_result = paper_inner
        self.px4_command = command
        self.command_computed_ns = now_ns
        self.control_reason = 'ok_paper_observer_control'

    def _controller_state(
        self,
        observer: ObserverState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the observer state consumed by the paper controller."""
        return (
            np.asarray(
                (observer.p_r.x, observer.p_r.y, observer.p_r.z),
                dtype=float,
            ),
            np.asarray(
                (observer.v_r.x, observer.v_r.y, observer.v_r.z),
                dtype=float,
            ),
            np.asarray(observer.image_xy, dtype=float),
        )

    def _update_visual_terminal(self, now_s: float) -> None:
        """Leave only after Gazebo contact and green confirmation."""
        if self.gate.phase != InterceptionPhase.ACTIVE:
            return
        if self.terminal_started_s is None and self.contact_confirmed:
            self.terminal_started_s = now_s
            self.visual_interception_complete = True
            self._latch_coast_setpoint()
            self.get_logger().warning(
                'Verified target contact: exiting paper interception control'
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
            self._update_visual_acquisition(now_s, now_ns)

        actions = self.gate.step(
            now_s=now_s,
            hover_ready=bool(
                self.bootstrap_sent
                and
                self.acquisition_command is not None
                and self.acquisition_command.ready
                and self._fresh_feature(now_ns) is not None
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
                f'reason={self.gate.terminal_reason or "visual_control"}; '
                f'control={self.control_reason}'
            )
            self.last_gate_phase = self.gate.phase
        if InterceptionAction.HOLD_POSITION in actions:
            self._publish_visual_acquisition_setpoint()
        elif InterceptionAction.STREAM_RATE_SETPOINT in actions:
            if self.gate.phase == InterceptionPhase.PRESTREAM:
                self._publish_rate_mode()
                self._publish_hover_rate_setpoint()
            else:
                self._publish_rate_mode()
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
        inner = self.inner_result
        if command is None:
            message.valid = False
            message.reason = (
                self.gate.terminal_reason or self.control_reason
            )
            self.debug_pub.publish(message)
            return
        if result is not None and inner is not None:
            message.z1 = result.z1
            _assign_vector(message.z2, result.z2_e)
            _assign_vector(message.acceleration_d, result.acceleration_d_e)
            message.attitude_d = tuple(result.attitude_d_b_to_e.reshape(9))
            _assign_vector(message.omega1_b, result.omega1_b)
            _assign_vector(message.omega2_b, inner.omega2_b)
            _assign_vector(message.omega_d_b, inner.omega_d_b)
            message.thrust_n = inner.thrust_n
            message.thrust_normalized = -command.thrust_body[2]
            message.thrust_saturated = inner.thrust_saturated
            message.thrust_mapping_saturated = command.thrust_saturated
            message.rate_saturated = inner.rate_saturated
            message.valid = True
            message.reason = (
                self.gate.terminal_reason or self.control_reason
            )
        else:
            message.valid = False
            message.reason = self.control_reason
        self.debug_pub.publish(message)

    def _publish_rate_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.body_rate = True
        self.offboard_pub.publish(message)

    def _update_visual_acquisition(self, now_s: float, now_ns: int) -> None:
        """Search for and safely acquire a fresh target while hovering."""
        heading = self.vehicle_heading_ned_rad
        if heading is None:
            return
        feature = self._fresh_feature(now_ns)
        x_norm = 0.0
        y_norm = 0.0
        if feature is not None:
            x_norm = float(feature.x_norm)
            y_norm = float(feature.y_norm)
        command = self.acquisition.step(
            now_s=now_s,
            current_yaw_rad=heading,
            target_detected=feature is not None,
            x_norm=x_norm,
            y_norm=y_norm,
        )
        self.acquisition_command = command
        if command.ready and not self.bootstrap_sent:
            if self.range_probe_started_s is None:
                self.range_probe_started_s = now_s
            bootstrap = self._range_bootstrap_message(now_ns)
            if bootstrap is not None:
                self._reset_observer_for_interception(bootstrap)
                self.range_probe_started_s = None
            else:
                reason = self.range_block_reason
                self.control_reason = f'waiting_for_range_bootstrap:{reason}'
                if reason != self.last_range_reason:
                    self.get_logger().info(self.control_reason)
                    self.last_range_reason = reason
        elif not command.ready:
            self.range_probe_started_s = None
        if command.phase != self.last_acquisition_phase:
            self.get_logger().warning(
                f'Visual acquisition transition: '
                f'{self.last_acquisition_phase.value} -> '
                f'{command.phase.value}'
            )
            self.last_acquisition_phase = command.phase

    def _fresh_feature(self, now_ns: int) -> Optional[VisionFeature]:
        """Return the current detection only while its receipt is fresh."""
        if self.feature is None or self.feature_received_ns is None:
            return None
        age_ns = now_ns - self.feature_received_ns
        timeout_ns = int(self.visual_feature_timeout_s * 1e9)
        if age_ns < 0 or age_ns > timeout_ns:
            return None
        return self.feature

    def _reset_observer_for_interception(
        self, bootstrap: ObserverBootstrap
    ) -> None:
        """Transfer a range estimate before defining interception x(0)."""
        self.observer_bootstrap_pub.publish(bootstrap)
        self.observer_reset_pub.publish(Empty())
        self.bootstrap_sent = True
        self.observer_state = None
        self.observer_received_ns = None
        self.visual_result = None
        self.inner_result = None
        self.px4_command = None
        self.command_computed_ns = None
        self.last_outer_ns = None
        self.contact_confirmed = False
        self.control_reason = 'waiting_for_interception_observer_reset'
        self.get_logger().warning(
            'Resetting observer with '
            f'{bootstrap.range_m:.2f} +/- {bootstrap.range_std_m:.2f} m '
            f'bootstrap from {bootstrap.sample_count} bearings'
        )

    def _publish_visual_acquisition_setpoint(self) -> None:
        """Hold position while search/alignment changes only yaw and height."""
        command = self.acquisition_command
        home_x = self.state_machine.home_x
        home_y = self.state_machine.home_y
        home_z = self.state_machine.home_z
        if (
            command is None
            or home_x is None
            or home_y is None
            or home_z is None
        ):
            return
        mode = OffboardControlMode()
        mode.timestamp = self._timestamp_us()
        mode.position = True
        self.offboard_pub.publish(mode)
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        # Acquisition may request a large vertical correction for a target
        # above or below the initial takeoff plane.  Clamp the actual PX4
        # setpoint to the same altitude envelope used by the flight state
        # machine, so a bad or lost image cannot command outside the fence.
        relative_altitude = _clip(
            self.state_machine.config.target_altitude_m
            - command.vertical_offset_m
            - self._range_probe_offset_m(),
            self.state_machine.config.min_hold_altitude_m,
            self.state_machine.config.max_relative_altitude_m,
        )
        setpoint.position = [home_x, home_y, home_z - relative_altitude]
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

    def _range_probe_offset_m(self) -> float:
        """Provide gentle vertical parallax only while visual lock is safe."""
        if self.range_probe_started_s is None or self.bootstrap_sent:
            return 0.0
        elapsed_s = max(
            0.0,
            self.get_clock().now().nanoseconds * 1e-9
            - self.range_probe_started_s,
        )
        return self.range_probe_amplitude_m * math.sin(
            2.0 * math.pi * elapsed_s / self.range_probe_period_s
        )

    def _publish_hover_rate_setpoint(self) -> None:
        command = adapt_rate_thrust_command(
            (0.0, 0.0, 0.0),
            self.paper_config.mass_kg * self.paper_config.gravity_m_s2,
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
        yaw_ned_rad = self.coast_yaw_ned_rad
        if yaw_ned_rad is None:
            yaw_ned_rad = (
                self.vehicle_heading_ned_rad
                if self.vehicle_heading_ned_rad is not None
                else self.target_yaw_rad
            )
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
        setpoint.yaw = float(yaw_ned_rad)
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
        timestamps = (
            self.feature_received_ns,
            self.observer_received_ns,
            self.odometry_received_ns,
        )
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
        return self.k_b - abs(self.visual_result.z1)


def main(args=None) -> None:
    """Run the integrated observer-fed paper controller."""
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
