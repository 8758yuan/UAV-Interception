"""Evaluate the P2 controller online without sending PX4 commands."""

import math
from typing import Optional

from interception_interfaces.msg import ControlDebug, RelativeState
import numpy as np
from px4_msgs.msg import VehicleOdometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_control.frames import px4_quaternion_to_enu_flu_rotation
from ibvs_control.so3_controller import (
    OuterLoopConfig,
    OuterLoopResult,
    attitude_rate_feedback,
    combine_and_saturate_rates,
    compute_outer_loop,
)
from ibvs_control.thrust_mapping import (
    ThrustMappingConfig,
    newtons_to_px4_normalized,
)


class ControllerShadow(Node):
    """Run 50 Hz outer and 200 Hz inner calculations in read-only mode."""

    def __init__(self) -> None:
        super().__init__('controller_shadow')
        self._declare_parameters()
        self.controller_config = OuterLoopConfig(
            k1=self._float_parameter('k1'),
            k2=self._float_parameter('k2'),
            k_b=1.0 - math.cos(
                math.radians(self._float_parameter('safe_los_angle_deg'))
            ),
            mass_kg=self._float_parameter('mass_kg'),
            thrust_max_n=self._float_parameter('thrust_max_n'),
            max_command_tilt_rad=math.radians(
                self._float_parameter('max_command_tilt_deg')
            ),
        )
        self.controller_config.validate()
        self.thrust_mapping_config = ThrustMappingConfig(
            mass_kg=self.controller_config.mass_kg,
            hover_thrust_normalized=self._float_parameter(
                'hover_thrust_normalized'
            ),
            gravity_m_s2=self.controller_config.gravity_m_s2,
        )
        self.thrust_mapping_config.validate()
        self.omega_max_rad_s = self._float_parameter('omega_max_rad_s')
        self.telemetry_timeout_s = self._float_parameter(
            'telemetry_timeout_s'
        )
        self.camera_axis_b = self._unit_parameter('camera_axis_b')
        outer_hz = self._float_parameter('outer_loop_hz')
        inner_hz = self._float_parameter('inner_loop_hz')
        if outer_hz <= 0.0 or inner_hz <= 0.0:
            raise ValueError('loop frequencies must be positive')

        self.relative_state: Optional[RelativeState] = None
        self.vehicle_odometry: Optional[VehicleOdometry] = None
        self.relative_received_ns: Optional[int] = None
        self.odometry_received_ns: Optional[int] = None
        self.outer_result: Optional[OuterLoopResult] = None
        self.outer_reason = 'waiting_for_telemetry'

        self.publisher = self.create_publisher(
            ControlDebug,
            str(self.get_parameter('debug_topic').value),
            10,
        )
        self.create_subscription(
            RelativeState,
            str(self.get_parameter('relative_state_topic').value),
            self._relative_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter('vehicle_odometry_topic').value),
            self._odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / outer_hz, self._outer_callback)
        self.create_timer(1.0 / inner_hz, self._inner_callback)
        self.get_logger().warning(
            'Controller shadow active: debug only; no PX4 inputs published'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('k1', 0.05)
        self.declare_parameter('k2', 20.0)
        self.declare_parameter('safe_los_angle_deg', 45.0)
        self.declare_parameter('mass_kg', 2.0)
        self.declare_parameter('thrust_max_n', 26.9784)
        self.declare_parameter('hover_thrust_normalized', 0.727)
        self.declare_parameter('omega_max_rad_s', 0.5)
        self.declare_parameter('max_command_tilt_deg', 20.0)
        self.declare_parameter('outer_loop_hz', 50.0)
        self.declare_parameter('inner_loop_hz', 200.0)
        self.declare_parameter('telemetry_timeout_s', 0.2)
        self.declare_parameter('camera_axis_b', [1.0, 0.0, 0.0])
        self.declare_parameter(
            'relative_state_topic',
            '/interception/truth/relative_state',
        )
        self.declare_parameter(
            'vehicle_odometry_topic',
            '/fmu/out/vehicle_odometry',
        )
        self.declare_parameter(
            'debug_topic',
            '/interception/control/debug',
        )

    def _relative_callback(self, msg: RelativeState) -> None:
        self.relative_state = msg
        self.relative_received_ns = self.get_clock().now().nanoseconds

    def _odometry_callback(self, msg: VehicleOdometry) -> None:
        self.vehicle_odometry = msg
        self.odometry_received_ns = self.get_clock().now().nanoseconds

    def _outer_callback(self) -> None:
        reason = self._telemetry_reason()
        if reason:
            self.outer_result = None
            self.outer_reason = reason
            return
        relative = self.relative_state
        odometry = self.vehicle_odometry
        try:
            attitude = px4_quaternion_to_enu_flu_rotation(odometry.q)
            designed_los = attitude @ self.camera_axis_b
            self.outer_result = compute_outer_loop(
                _message_vector(relative.p_r),
                _message_vector(relative.v_r),
                _message_vector(relative.los),
                designed_los,
                attitude,
                self.controller_config,
            )
            self.outer_reason = 'ok'
        except ValueError as error:
            self.outer_result = None
            self.outer_reason = str(error)

    def _inner_callback(self) -> None:
        message = ControlDebug()
        message.stamp = self.get_clock().now().to_msg()
        outer = self.outer_result
        if outer is None:
            message.valid = False
            message.reason = self.outer_reason
            self.publisher.publish(message)
            return
        try:
            attitude = px4_quaternion_to_enu_flu_rotation(
                self.vehicle_odometry.q
            )
            omega2_b = attitude_rate_feedback(
                outer.attitude_d_b_to_e,
                attitude,
            )
            raw_rate = outer.omega1_b + omega2_b
            omega_d_b = combine_and_saturate_rates(
                outer.omega1_b,
                omega2_b,
                self.omega_max_rad_s,
            )
        except ValueError as error:
            message.valid = False
            message.reason = str(error)
            self.publisher.publish(message)
            return

        message.z1 = outer.z1
        _assign_vector(message.z2, outer.z2_e)
        _assign_vector(message.acceleration_d, outer.acceleration_d_e)
        message.attitude_d = tuple(outer.attitude_d_b_to_e.reshape(9))
        _assign_vector(message.omega1_b, outer.omega1_b)
        _assign_vector(message.omega2_b, omega2_b)
        _assign_vector(message.omega_d_b, omega_d_b)
        message.thrust_n = outer.thrust_n
        normalized_thrust = newtons_to_px4_normalized(
            outer.thrust_n,
            self.thrust_mapping_config,
        )
        message.thrust_normalized = normalized_thrust.magnitude
        message.thrust_saturated = outer.thrust_saturated
        message.thrust_mapping_saturated = normalized_thrust.saturated
        message.rate_saturated = bool(
            np.linalg.norm(raw_rate) > self.omega_max_rad_s
        )
        message.valid = True
        message.reason = 'ok'
        self.publisher.publish(message)

    def _telemetry_reason(self) -> str:
        now_ns = self.get_clock().now().nanoseconds
        telemetry = (
            ('relative_state', self.relative_state, self.relative_received_ns),
            (
                'vehicle_odometry',
                self.vehicle_odometry,
                self.odometry_received_ns,
            ),
        )
        for name, value, received_ns in telemetry:
            if value is None or received_ns is None:
                return f'waiting_for_{name}'
            if (now_ns - received_ns) * 1e-9 > self.telemetry_timeout_s:
                return f'{name}_timeout'
        if self.relative_state.source != 'truth':
            return 'relative_state_source_is_not_truth'
        if (
            self.vehicle_odometry.pose_frame
            != VehicleOdometry.POSE_FRAME_NED
        ):
            return 'vehicle_pose_frame_is_not_ned'
        return ''

    def _float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        return value

    def _unit_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must be a finite three-vector')
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError(f'{name} must be nonzero')
        return value / norm


def _message_vector(message) -> tuple:
    return message.x, message.y, message.z


def _assign_vector(message, values) -> None:
    message.x, message.y, message.z = (float(value) for value in values)


def main(args=None) -> None:
    """Run the read-only P2 controller shadow."""
    rclpy.init(args=args)
    node = ControllerShadow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
