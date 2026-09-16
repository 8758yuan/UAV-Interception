"""Pure P2 relative-state calculations in the ROS ENU world frame."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RelativeKinematics:
    """Truth kinematics using the paper's relative-vector convention."""

    p_r: Vector3
    v_r: Vector3
    los: Vector3
    distance_m: float


@dataclass(frozen=True)
class ClosestApproach:
    """Current result accumulated by :class:`ClosestApproachTracker`."""

    d_min_m: float
    closest_time_s: float
    hit: bool
    intercept_time_s: Optional[float]


def compute_relative_kinematics(
    interceptor_position_e: Sequence[float],
    interceptor_velocity_e: Sequence[float],
    target_position_e: Sequence[float],
    target_velocity_e: Sequence[float],
    *,
    minimum_distance_m: float = 1e-6,
) -> RelativeKinematics:
    """Return ``p_r = p_I - p_T``, ``v_r`` and interceptor-to-target LOS."""
    p_i = _vector3(interceptor_position_e, 'interceptor_position_e')
    v_i = _vector3(interceptor_velocity_e, 'interceptor_velocity_e')
    p_t = _vector3(target_position_e, 'target_position_e')
    v_t = _vector3(target_velocity_e, 'target_velocity_e')
    if not math.isfinite(minimum_distance_m) or minimum_distance_m <= 0.0:
        raise ValueError('minimum_distance_m must be finite and positive')

    p_r = tuple(p_i[index] - p_t[index] for index in range(3))
    v_r = tuple(v_i[index] - v_t[index] for index in range(3))
    distance_m = math.sqrt(sum(value * value for value in p_r))
    if distance_m < minimum_distance_m:
        raise ValueError('line of sight is undefined at zero relative distance')
    los = tuple(-value / distance_m for value in p_r)
    return RelativeKinematics(p_r, v_r, los, distance_m)


class ClosestApproachTracker:
    """Accumulate minimum distance and first hit time for one P2 trial."""

    def __init__(self, hit_radius_m: float = 0.5) -> None:
        if not math.isfinite(hit_radius_m) or hit_radius_m <= 0.0:
            raise ValueError('hit_radius_m must be finite and positive')
        self.hit_radius_m = hit_radius_m
        self._last_time_s: Optional[float] = None
        self._d_min_m = math.inf
        self._closest_time_s = math.nan
        self._intercept_time_s: Optional[float] = None

    def update(self, time_s: float, distance_m: float) -> ClosestApproach:
        """Add a monotonic finite sample and return the updated metrics."""
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError('time_s must be finite and nonnegative')
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError('time_s must be monotonic')
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError('distance_m must be finite and nonnegative')
        self._last_time_s = time_s
        if distance_m < self._d_min_m:
            self._d_min_m = distance_m
            self._closest_time_s = time_s
        if (
            self._intercept_time_s is None
            and distance_m <= self.hit_radius_m
        ):
            self._intercept_time_s = time_s
        return self.result

    @property
    def result(self) -> ClosestApproach:
        """Return the current immutable metric snapshot."""
        return ClosestApproach(
            d_min_m=self._d_min_m,
            closest_time_s=self._closest_time_s,
            hit=self._intercept_time_s is not None,
            intercept_time_s=self._intercept_time_s,
        )


def _vector3(values: Sequence[float], name: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f'{name} must contain exactly three values')
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f'{name} values must be finite')
    return vector
