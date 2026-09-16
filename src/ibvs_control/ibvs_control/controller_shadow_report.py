"""Record a bounded controller-shadow observation and emit JSON metrics."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from interception_interfaces.msg import ControlDebug
from px4_msgs.msg import VehicleLandDetected
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_control.shadow_metrics import ShadowMetrics


class ControllerShadowReport(Node):
    """Summarize a fixed amount of simulated-time shadow debug data."""

    def __init__(self) -> None:
        super().__init__('controller_shadow_report')
        self.declare_parameter('duration_s', 30.0)
        self.declare_parameter('safe_los_angle_deg', 45.0)
        self.declare_parameter('trial_id', 'p2_shadow_01')
        self.declare_parameter('output_directory', 'results/p2/shadow')
        self.declare_parameter('debug_topic', '/interception/control/debug')
        self.declare_parameter(
            'vehicle_land_detected_topic',
            '/fmu/out/vehicle_land_detected',
        )
        self.declare_parameter('require_airborne', True)
        self.declare_parameter('airborne_settle_s', 12.0)
        self.duration_target_s = float(self.get_parameter('duration_s').value)
        safe_angle_deg = float(
            self.get_parameter('safe_los_angle_deg').value
        )
        if not math.isfinite(self.duration_target_s) or self.duration_target_s <= 0:
            raise ValueError('duration_s must be finite and positive')
        if not math.isfinite(safe_angle_deg) or not 0.0 < safe_angle_deg < 180.0:
            raise ValueError('safe_los_angle_deg must be within (0, 180)')
        self.metrics = ShadowMetrics(
            k_b=1.0 - math.cos(math.radians(safe_angle_deg))
        )
        self.finished = False
        self.require_airborne = bool(
            self.get_parameter('require_airborne').value
        )
        self.airborne_settle_s = float(
            self.get_parameter('airborne_settle_s').value
        )
        if (
            not math.isfinite(self.airborne_settle_s)
            or self.airborne_settle_s < 0.0
        ):
            raise ValueError('airborne_settle_s must be finite and nonnegative')
        self.landed = None
        self.airborne_since_s = None
        self.output_path = None
        self.create_subscription(
            ControlDebug,
            str(self.get_parameter('debug_topic').value),
            self._callback,
            100,
        )
        self.create_subscription(
            VehicleLandDetected,
            str(self.get_parameter('vehicle_land_detected_topic').value),
            self._landed_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'Recording {self.duration_target_s:.1f} s of shadow debug data; '
            f'require_airborne={self.require_airborne}'
        )

    def _landed_callback(self, message: VehicleLandDetected) -> None:
        self.landed = bool(message.landed)

    def _callback(self, message: ControlDebug) -> None:
        if self.finished:
            return
        stamp_s = message.stamp.sec + message.stamp.nanosec * 1e-9
        if self.require_airborne:
            if self.landed is not False:
                self.airborne_since_s = None
                return
            if self.airborne_since_s is None:
                self.airborne_since_s = stamp_s
                return
            if stamp_s - self.airborne_since_s < self.airborne_settle_s:
                return
        try:
            self.metrics.update(
                stamp_s=stamp_s,
                valid=message.valid,
                reason=message.reason,
                z1=message.z1,
                thrust_n=message.thrust_n,
                thrust_normalized=message.thrust_normalized,
                thrust_saturated=message.thrust_saturated,
                thrust_mapping_saturated=message.thrust_mapping_saturated,
                rate_saturated=message.rate_saturated,
            )
        except ValueError as error:
            self.get_logger().error(f'Invalid debug sample: {error}')
            return
        if self.metrics.duration_s >= self.duration_target_s:
            self._write_report()

    def _write_report(self) -> None:
        summary = self.metrics.summary().to_dict()
        summary.update(
            {
                'trial_id': str(self.get_parameter('trial_id').value),
                'target_duration_s': self.duration_target_s,
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'control_mode': 'shadow_no_px4_output',
                'capture_condition': (
                    'airborne_after_settle_time'
                    if self.require_airborne
                    else 'airborne_not_required'
                ),
                'airborne_settle_s': self.airborne_settle_s,
            }
        )
        output_directory = Path(
            str(self.get_parameter('output_directory').value)
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        self.output_path = output_directory / f"{summary['trial_id']}.json"
        self.output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        self.finished = True
        self.get_logger().info(f'Shadow report written: {self.output_path}')


def main(args=None) -> None:
    """Run until the configured shadow observation is complete."""
    rclpy.init(args=args)
    node = ControllerShadowReport()
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
