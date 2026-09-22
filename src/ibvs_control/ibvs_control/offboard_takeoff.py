"""
PX4 Offboard takeoff, hold, and landing node for the P1 flight-interface test.

The node streams local NED position setpoints while a feedback-driven state
machine requests Offboard mode and arming.  It automatically aborts to a PX4
land command after telemetry loss, failsafe, invalid data, excessive tilt, or
geofence violations.
"""

import math
from typing import Dict

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from ibvs_control.frames import tilt_from_body_to_ned_quaternion
from ibvs_control.takeoff_state_machine import (
    FlightAction,
    FlightPhase,
    TakeoffConfig,
    TakeoffStateMachine,
)


class OffboardTakeoff(Node):
    """Execute a feedback-driven PX4 takeoff, hold, and landing sequence."""

    def __init__(
        self,
        node_name: str = 'offboard_takeoff',
        auto_land_default: bool = True,
        command_output_default: bool = True,
    ) -> None:
        super().__init__(node_name)
        self._declare_parameters(auto_land_default)
        self.declare_parameter(
            'enable_flight_commands',
            command_output_default,
        )
        self.command_output_enabled = self._bool_parameter(
            'enable_flight_commands'
        )

        self.target_yaw_rad = self._float_parameter('target_yaw_rad')
        self.target_system = self._int_parameter('target_system')
        self.target_component = self._int_parameter('target_component')
        self.source_system = self._int_parameter('source_system')
        self.source_component = self._int_parameter('source_component')
        self.stop_on_complete = self._bool_parameter('stop_on_complete')

        config = TakeoffConfig(
            target_altitude_m=self._float_parameter('target_altitude_m'),
            altitude_tolerance_m=self._float_parameter(
                'altitude_tolerance_m'
            ),
            vertical_speed_tolerance_m_s=self._float_parameter(
                'vertical_speed_tolerance_m_s'
            ),
            altitude_settle_time_s=self._float_parameter(
                'altitude_settle_time_s'
            ),
            prestream_duration_s=self._float_parameter(
                'prestream_duration_s'
            ),
            command_retry_period_s=self._float_parameter(
                'command_retry_period_s'
            ),
            init_timeout_s=self._float_parameter('init_timeout_s'),
            offboard_timeout_s=self._float_parameter('offboard_timeout_s'),
            arm_timeout_s=self._float_parameter('arm_timeout_s'),
            takeoff_timeout_s=self._float_parameter('takeoff_timeout_s'),
            hold_duration_s=self._float_parameter('hold_duration_s'),
            land_timeout_s=self._float_parameter('land_timeout_s'),
            telemetry_timeout_s=self._float_parameter('telemetry_timeout_s'),
            max_horizontal_distance_m=self._float_parameter(
                'max_horizontal_distance_m'
            ),
            max_relative_altitude_m=self._float_parameter(
                'max_relative_altitude_m'
            ),
            min_hold_altitude_m=self._float_parameter(
                'min_hold_altitude_m'
            ),
            max_tilt_rad=math.radians(
                self._float_parameter('max_tilt_deg')
            ),
            auto_land=self._bool_parameter('auto_land'),
        )
        self.state_machine = TakeoffStateMachine(
            config,
            # With use_sim_time, the clock is zero until the first /clock
            # sample.  Start INIT timeout timing in the first timer callback
            # so that the initial Gazebo clock jump cannot cause an ABORT.
            start_time_s=None,
        )
        self._last_logged_phase = self.state_machine.phase

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

        self.offboard_pub = None
        self.trajectory_pub = None
        self.command_pub = None
        if self.command_output_enabled:
            self.offboard_pub = self.create_publisher(
                OffboardControlMode,
                self._topic('topics.offboard_control_mode'),
                qos_pub,
            )
            self.trajectory_pub = self.create_publisher(
                TrajectorySetpoint,
                self._topic('topics.trajectory_setpoint'),
                qos_pub,
            )
            self.command_pub = self.create_publisher(
                VehicleCommand,
                self._topic('topics.vehicle_command'),
                qos_pub,
            )
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            self._topic('topics.vehicle_local_position'),
            self.position_callback,
            qos_sub,
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            self._topic('topics.vehicle_status'),
            self.status_callback,
            qos_sub,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            self._topic('topics.vehicle_attitude'),
            self.attitude_callback,
            qos_sub,
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            self._topic('topics.vehicle_land_detected'),
            self.land_detected_callback,
            qos_sub,
        )
        self.command_ack_sub = self.create_subscription(
            VehicleCommandAck,
            self._topic('topics.vehicle_command_ack'),
            self.command_ack_callback,
            qos_sub,
        )

        timer_period_s = self._float_parameter('timer_period_s')
        if timer_period_s <= 0.0:
            raise ValueError('timer_period_s must be greater than zero')
        self.timer = self.create_timer(timer_period_s, self.timer_callback)

        self.get_logger().info(
            'P1 state machine initialized; waiting for status, local position, '
            'and attitude'
        )

    def _declare_parameters(self, auto_land_default: bool) -> None:
        topic_defaults: Dict[str, str] = {
            'offboard_control_mode': '/fmu/in/offboard_control_mode',
            'trajectory_setpoint': '/fmu/in/trajectory_setpoint',
            'vehicle_command': '/fmu/in/vehicle_command',
            'vehicle_local_position': '/fmu/out/vehicle_local_position_v1',
            'vehicle_status': '/fmu/out/vehicle_status_v4',
            'vehicle_attitude': '/fmu/out/vehicle_attitude',
            'vehicle_land_detected': '/fmu/out/vehicle_land_detected',
            'vehicle_command_ack': '/fmu/out/vehicle_command_ack_v1',
        }
        for name, default in topic_defaults.items():
            self.declare_parameter(f'topics.{name}', default)

        numeric_defaults = {
            'timer_period_s': 0.1,
            'target_altitude_m': 2.0,
            'target_yaw_rad': 0.0,
            'altitude_tolerance_m': 0.15,
            'vertical_speed_tolerance_m_s': 0.30,
            'altitude_settle_time_s': 1.0,
            'prestream_duration_s': 2.0,
            'command_retry_period_s': 1.0,
            'init_timeout_s': 15.0,
            'offboard_timeout_s': 10.0,
            'arm_timeout_s': 10.0,
            'takeoff_timeout_s': 30.0,
            'hold_duration_s': 10.0,
            'land_timeout_s': 30.0,
            'telemetry_timeout_s': 1.0,
            'max_horizontal_distance_m': 3.0,
            # 起飞/搜索阶段相对home点允许的最大高度，单位m。
            'max_relative_altitude_m': 50.0,
            'min_hold_altitude_m': 0.5,
            'max_tilt_deg': 35.0,
        }
        for name, default in numeric_defaults.items():
            self.declare_parameter(name, default)

        self.declare_parameter('auto_land', auto_land_default)
        self.declare_parameter('stop_on_complete', False)
        self.declare_parameter('target_system', 1)
        self.declare_parameter('target_component', 1)
        self.declare_parameter('source_system', 1)
        self.declare_parameter('source_component', 1)

    def _topic(self, parameter_name: str) -> str:
        topic = str(self.get_parameter(parameter_name).value)
        if not topic:
            raise ValueError(f'{parameter_name} must not be empty')
        return topic

    def _float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        return value

    def _int_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool_parameter(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def _timestamp_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def position_callback(self, msg: VehicleLocalPosition) -> None:
        """Forward valid local NED position feedback to the state machine."""
        self.state_machine.update_position(
            x=float(msg.x),
            y=float(msg.y),
            z=float(msg.z),
            vz=float(msg.vz),
            valid=bool(msg.xy_valid and msg.z_valid and msg.v_z_valid),
            now_s=self._now_s(),
        )

    def status_callback(self, msg: VehicleStatus) -> None:
        """Use PX4's reported state as the authority for transitions."""
        self.state_machine.update_status(
            armed=msg.arming_state == VehicleStatus.ARMING_STATE_ARMED,
            offboard=msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD,
            auto_land_active=(
                msg.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
            ),
            preflight_ok=bool(msg.pre_flight_checks_pass),
            failsafe=bool(msg.failsafe),
            now_s=self._now_s(),
        )

    def attitude_callback(self, msg: VehicleAttitude) -> None:
        """Compute tilt from PX4's body-FRD to earth-NED quaternion."""
        try:
            tilt_rad = tilt_from_body_to_ned_quaternion(msg.q)
        except ValueError:
            self.state_machine.update_attitude(
                tilt_rad=0.0,
                valid=False,
                now_s=self._now_s(),
            )
            return

        self.state_machine.update_attitude(
            tilt_rad=tilt_rad,
            valid=True,
            now_s=self._now_s(),
        )

    def land_detected_callback(self, msg: VehicleLandDetected) -> None:
        """Track PX4's authoritative landed state."""
        self.state_machine.update_landed(bool(msg.landed))

    def command_ack_callback(self, msg: VehicleCommandAck) -> None:
        """Log command results and abort after permanent command rejection."""
        command_names = {
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE: 'OFFBOARD',
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM: 'ARM',
            VehicleCommand.VEHICLE_CMD_NAV_LAND: 'LAND',
        }
        command_name = command_names.get(msg.command)
        if command_name is None:
            return

        accepted_results = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS,
        }
        retry_results = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED,
        }
        if msg.result in accepted_results:
            self.get_logger().info(f'{command_name} command acknowledged')
        elif msg.result in retry_results:
            self.get_logger().warning(
                f'{command_name} command temporarily rejected; will retry'
            )
        else:
            reason = f'{command_name.lower()}_command_rejected_{msg.result}'
            self.get_logger().error(reason)
            self.state_machine.abort(reason, self._now_s())

    def publish_offboard_control_mode(self) -> None:
        """Select position control as the active Offboard command type."""
        if self.offboard_pub is None:
            return
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self) -> None:
        """Hold initial horizontal position at the configured relative height."""
        if self.trajectory_pub is None:
            return
        target_z = self.state_machine.target_z_ned_m
        home_x = self.state_machine.home_x
        home_y = self.state_machine.home_y
        if target_z is None or home_x is None or home_y is None:
            return

        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position = [home_x, home_y, target_z]
        msg.yaw = self.target_yaw_rad
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        """Publish a command addressed to the configured PX4 vehicle."""
        if self.command_pub is None:
            return
        msg = VehicleCommand()
        msg.timestamp = self._timestamp_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.target_system
        msg.target_component = self.target_component
        msg.source_system = self.source_system
        msg.source_component = self.source_component
        msg.confirmation = 0
        msg.from_external = True
        self.command_pub.publish(msg)

    def timer_callback(self) -> None:
        """Advance the mission and execute the requested PX4 actions."""
        if not self.command_output_enabled:
            return
        actions = self.state_machine.step(self._now_s())
        self._log_phase_change()
        self._execute_actions(actions)

        if (
            self.stop_on_complete
            and self.state_machine.phase == FlightPhase.COMPLETE
        ):
            self.get_logger().info('Mission complete; stopping timer')
            self.timer.cancel()

    def _execute_actions(self, actions) -> None:
        """Translate pure state-machine actions into PX4 publications."""
        for action in actions:
            if action == FlightAction.STREAM_POSITION_SETPOINT:
                self.publish_offboard_control_mode()
                self.publish_trajectory_setpoint()
            elif action == FlightAction.REQUEST_OFFBOARD:
                self.get_logger().info('Requesting OFFBOARD mode')
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    1.0,
                    6.0,
                )
            elif action == FlightAction.REQUEST_ARM:
                self.get_logger().info('Requesting vehicle arm')
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    float(VehicleCommand.ARMING_ACTION_ARM),
                )
            elif action == FlightAction.REQUEST_LAND:
                self.get_logger().warning('Requesting automatic landing')
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_NAV_LAND
                )

    def _log_phase_change(self) -> None:
        phase = self.state_machine.phase
        if phase == self._last_logged_phase:
            return

        previous = self._last_logged_phase
        self._last_logged_phase = phase
        message = f'State transition: {previous.value} -> {phase.value}'
        if phase == FlightPhase.ABORT:
            self.get_logger().error(
                f'{message}; reason={self.state_machine.abort_reason}'
            )
        else:
            self.get_logger().info(message)


def main(args=None) -> None:
    """Run the feedback-driven Offboard takeoff node."""
    rclpy.init(args=args)
    node = OffboardTakeoff()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
