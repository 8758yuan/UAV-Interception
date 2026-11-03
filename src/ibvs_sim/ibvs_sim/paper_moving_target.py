"""Paper target trajectory used to check the shared Gazebo physics driver."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Tuple



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
        tilt_deg: float = 0.0,
    ) -> None:
        if pattern not in ('figure8', 'circle'):
            raise ValueError("pattern must be 'figure8' or 'circle'")
        values = (*origin, speed_m_s, radius_x_m, radius_y_m, tilt_deg)
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
        self._tilt_rad = math.radians(tilt_deg)
        self._samples = int(samples)
        self._theta_step = 2.0 * math.pi / self._samples
        self._arc_lengths = self._build_arc_length_table()
        self.path_length_m = self._arc_lengths[-1]

    def sample(self, elapsed_s: float) -> TrajectorySample:
        """Return a sample whose spatial speed equals the configured speed."""
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
        cos_tilt = math.cos(self._tilt_rad)
        sin_tilt = math.sin(self._tilt_rad)
        origin_x, origin_y, origin_z = self.origin
        return TrajectorySample(
            position=(
                origin_x + local_x * cos_tilt,
                origin_y + local_y,
                origin_z + local_x * sin_tilt,
            ),
            velocity=(velocity_x * cos_tilt, velocity_y, velocity_x * sin_tilt),
            yaw_rad=math.atan2(velocity_y, velocity_x * cos_tilt),
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
