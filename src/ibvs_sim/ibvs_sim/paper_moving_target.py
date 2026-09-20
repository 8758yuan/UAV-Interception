"""Drive a Gazebo-only moving target without exposing target truth to ROS."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Tuple

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GazeboNode
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


@dataclass(frozen=True)
class TrajectorySample:
    """One target pose and velocity sample in Gazebo's ENU world frame."""

    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    yaw_rad: float


class PaperTargetTrajectory:
    """Constant-arc-speed figure-eight or circular target trajectory."""

    def __init__(
        self,
        pattern: str,
        origin: Tuple[float, float, float],
        speed_m_s: float,
        radius_x_m: float,
        radius_y_m: float,
        samples: int = 8192,
    ) -> None:
        if pattern not in ('figure8', 'circle'):
            raise ValueError("pattern must be 'figure8' or 'circle'")
        values = (*origin, speed_m_s, radius_x_m, radius_y_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('trajectory parameters must be finite')
        if speed_m_s <= 0.0 or radius_x_m <= 0.0 or radius_y_m <= 0.0:
            raise ValueError('speed and radii must be positive')
        if samples < 128:
            raise ValueError('trajectory samples must be at least 128')
        self.pattern = pattern
        self.origin = tuple(float(value) for value in origin)
        self.speed_m_s = float(speed_m_s)
        self.radius_x_m = float(radius_x_m)
        self.radius_y_m = float(radius_y_m)
        self._samples = int(samples)
        self._theta_step = 2.0 * math.pi / self._samples
        self._arc_lengths = self._build_arc_length_table()
        self.path_length_m = self._arc_lengths[-1]

    def sample(self, elapsed_s: float) -> TrajectorySample:
        """Return a sample whose planar speed equals the configured speed."""
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError('elapsed_s must be finite and nonnegative')
        distance = math.fmod(
            self.speed_m_s * elapsed_s,
            self.path_length_m,
        )
        index = min(
            bisect_right(self._arc_lengths, distance) - 1,
            self._samples - 1,
        )
        segment_start = self._arc_lengths[index]
        segment_length = self._arc_lengths[index + 1] - segment_start
        fraction = (
            (distance - segment_start) / segment_length
            if segment_length > 0.0 else 0.0
        )
        theta = (index + fraction) * self._theta_step
        local_x, local_y = self._curve(theta)
        derivative_x, derivative_y = self._curve_derivative(theta)
        derivative_norm = math.hypot(derivative_x, derivative_y)
        velocity_x = self.speed_m_s * derivative_x / derivative_norm
        velocity_y = self.speed_m_s * derivative_y / derivative_norm
        origin_x, origin_y, origin_z = self.origin
        return TrajectorySample(
            position=(origin_x + local_x, origin_y + local_y, origin_z),
            velocity=(velocity_x, velocity_y, 0.0),
            yaw_rad=math.atan2(velocity_y, velocity_x),
        )

    def _curve(self, theta: float) -> Tuple[float, float]:
        if self.pattern == 'figure8':
            return (
                self.radius_x_m * math.sin(theta),
                self.radius_y_m * math.sin(2.0 * theta),
            )
        # Offset the circle so theta=0 starts at the configured spawn point.
        return (
            self.radius_x_m * (math.cos(theta) - 1.0),
            self.radius_y_m * math.sin(theta),
        )

    def _curve_derivative(self, theta: float) -> Tuple[float, float]:
        if self.pattern == 'figure8':
            return (
                self.radius_x_m * math.cos(theta),
                2.0 * self.radius_y_m * math.cos(2.0 * theta),
            )
        return (
            -self.radius_x_m * math.sin(theta),
            self.radius_y_m * math.cos(theta),
        )

    def _build_arc_length_table(self) -> Tuple[float, ...]:
        cumulative = [0.0]
        previous_x, previous_y = self._curve(0.0)
        for index in range(1, self._samples + 1):
            theta = index * self._theta_step
            current_x, current_y = self._curve(theta)
            cumulative.append(
                cumulative[-1]
                + math.hypot(
                    current_x - previous_x,
                    current_y - previous_y,
                )
            )
            previous_x, previous_y = current_x, current_y
        return tuple(cumulative)


class PaperMovingTarget(Node):
    """Apply the paper target trajectory only inside the Gazebo plant."""

    def __init__(self) -> None:
        super().__init__('paper_moving_target')
        self.declare_parameter('world_name', 'default')
        self.declare_parameter('model_name', 'ibvs_target')
        self.declare_parameter('pattern', 'figure8')
        self.declare_parameter('origin_x_m', 12.0)
        self.declare_parameter('origin_y_m', 0.0)
        self.declare_parameter('origin_z_m', 3.0)
        self.declare_parameter('speed_m_s', 5.0)
        self.declare_parameter('radius_x_m', 4.0)
        self.declare_parameter('radius_y_m', 2.0)
        self.declare_parameter('update_rate_hz', 100.0)
        self.declare_parameter('start_on_observer_reset', True)
        self.declare_parameter(
            'start_topic',
            '/interception/observer/reset',
        )
        self.declare_parameter(
            'contact_confirmation_topic',
            '/interception/target/green_confirmed',
        )
        update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        if not math.isfinite(update_rate_hz) or update_rate_hz <= 0.0:
            raise ValueError('update_rate_hz must be finite and positive')
        self.trajectory = PaperTargetTrajectory(
            pattern=str(self.get_parameter('pattern').value),
            origin=(
                float(self.get_parameter('origin_x_m').value),
                float(self.get_parameter('origin_y_m').value),
                float(self.get_parameter('origin_z_m').value),
            ),
            speed_m_s=float(self.get_parameter('speed_m_s').value),
            radius_x_m=float(self.get_parameter('radius_x_m').value),
            radius_y_m=float(self.get_parameter('radius_y_m').value),
        )
        world_name = str(self.get_parameter('world_name').value)
        self.model_name = str(self.get_parameter('model_name').value)
        if not world_name or not self.model_name:
            raise ValueError('world_name and model_name must not be empty')
        self.set_pose_service = f'/world/{world_name}/set_pose'
        self.gz_node = GazeboNode()
        self.start_time_ns = None
        self.stopped = False
        self.failure_count = 0
        self.start_on_reset = bool(
            self.get_parameter('start_on_observer_reset').value
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter('start_topic').value),
            self._start_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('contact_confirmation_topic').value),
            self._contact_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / update_rate_hz, self._update_pose)
        if not self.start_on_reset:
            self.start_time_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f'Target motion ready: {self.trajectory.pattern}, '
            f'{self.trajectory.speed_m_s:.1f} m/s; Gazebo-only set_pose'
        )

    def _start_callback(self, _message: Empty) -> None:
        if self.start_time_ns is None and not self.stopped:
            self.start_time_ns = self.get_clock().now().nanoseconds
            self.get_logger().warning('Paper moving-target trajectory started')

    def _contact_callback(self, message: Bool) -> None:
        if message.data:
            # Stop teleporting after contact so the lightweight target can
            # yield naturally and the verified post-impact sequence can run.
            self.stopped = True
            self.get_logger().warning('Target trajectory stopped after contact')

    def _update_pose(self) -> None:
        if self.stopped or self.start_time_ns is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        elapsed_s = max(0.0, (now_ns - self.start_time_ns) * 1e-9)
        sample = self.trajectory.sample(elapsed_s)
        request = _pose_request(self.model_name, sample)
        executed, response = self.gz_node.request(
            self.set_pose_service,
            request,
            Pose,
            Boolean,
            100,
        )
        if executed and response.data:
            self.failure_count = 0
            return
        self.failure_count += 1
        if self.failure_count in (1, 10, 50):
            self.get_logger().error(
                f'Gazebo set_pose failed ({self.failure_count} consecutive)'
            )


def _pose_request(model_name: str, sample: TrajectorySample) -> Pose:
    """Build a Gazebo pose command without creating a ROS truth topic."""
    if not model_name:
        raise ValueError('model_name must not be empty')
    request = Pose()
    request.name = model_name
    request.position.x, request.position.y, request.position.z = sample.position
    half_yaw = 0.5 * sample.yaw_rad
    request.orientation.z = math.sin(half_yaw)
    request.orientation.w = math.cos(half_yaw)
    return request


def main(args=None) -> None:
    """Run the Gazebo-only paper moving-target driver."""
    rclpy.init(args=args)
    node = PaperMovingTarget()
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
