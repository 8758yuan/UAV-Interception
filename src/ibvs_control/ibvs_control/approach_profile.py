"""Continuous radial-speed targets and rolling quintic distance references."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


_ENDPOINT_INVERSE = np.array((
    (10.0, -4.0, 0.5), (-15.0, 7.0, -1.0), (6.0, -3.0, 0.5),
))


@dataclass(frozen=True)
class ApproachProfileConfig:
    """Design parameters from the V3 approach-planning proposal."""

    horizon_s: float = 2.0
    replan_hz: float = 5.0
    alpha: float = 10.0
    beta: float = 0.1
    delta_v_m_s: float = 0.0
    boost_length_m: float = 10.0
    terminal_length_m: float = 2.0
    terminal_speed_m_s: float = 0.45

    def validate(self) -> None:
        """Check numerical domains, without imposing flight constraints."""
        positive = (
            self.horizon_s, self.replan_hz, self.boost_length_m,
            self.terminal_length_m, self.terminal_speed_m_s,
        )
        nonnegative = (self.alpha, self.beta, self.delta_v_m_s)
        if not all(math.isfinite(x) and x > 0.0 for x in positive):
            raise ValueError(
                'approach times, scales and terminal speed must be positive'
            )
        if not all(math.isfinite(x) and x >= 0.0 for x in nonnegative):
            raise ValueError(
                'approach alpha, beta and delta_v must be nonnegative'
            )


@dataclass(frozen=True)
class ApproachReference:
    """Distance, closing speed and its derivative at one reference instant."""

    distance_m: float
    speed_m_s: float
    acceleration_m_s2: float


@dataclass(frozen=True)
class QuinticApproachSegment:
    """One distance polynomial in normalized time s = (t - t0) / H."""

    start_time_s: float
    horizon_s: float
    coefficients: np.ndarray
    start: ApproachReference
    end: ApproachReference

    @classmethod
    def from_boundaries(
        cls,
        start_time_s: float,
        horizon_s: float,
        start: ApproachReference,
        end: ApproachReference,
    ) -> 'QuinticApproachSegment':
        """Solve all six distance/closing-speed/acceleration conditions."""
        values = (
            start_time_s, horizon_s, start.distance_m, start.speed_m_s,
            start.acceleration_m_s2, end.distance_m, end.speed_m_s,
            end.acceleration_m_s2,
        )
        if not all(math.isfinite(x) for x in values) or horizon_s <= 0.0:
            raise ValueError(
                'quintic boundaries must be finite and horizon positive'
            )
        b0 = start.distance_m
        b1 = -horizon_s * start.speed_m_s
        b2 = -horizon_s**2 * start.acceleration_m_s2 / 2.0
        residual = np.array((
            end.distance_m - b0 - b1 - b2,
            -horizon_s * end.speed_m_s - b1 - 2.0 * b2,
            -horizon_s**2 * end.acceleration_m_s2 - 2.0 * b2,
        ))
        # Inverse of the fixed endpoint matrix in proposal section 5.2.
        b3, b4, b5 = _ENDPOINT_INVERSE @ residual
        return cls(
            start_time_s, horizon_s,
            np.array((b0, b1, b2, b3, b4, b5)), start, end,
        )

    def sample(self, time_s: float) -> ApproachReference:
        """Sample the finite segment or retain its completed endpoint."""
        if not math.isfinite(time_s):
            raise ValueError('reference time must be finite')
        s = (time_s - self.start_time_s) / self.horizon_s
        if s <= 0.0:
            return self.start
        if s >= 1.0:
            return self.end
        b = self.coefficients
        return ApproachReference(
            distance_m=float(np.polynomial.polynomial.polyval(s, b)),
            speed_m_s=-float(np.polynomial.polynomial.polyval(
                s, np.arange(1, 6) * b[1:]
            )) / self.horizon_s,
            acceleration_m_s2=-float(np.polynomial.polynomial.polyval(
                s, np.arange(2, 6) * np.arange(1, 5) * b[2:]
            )) / self.horizon_s**2,
        )


class ApproachPlanner:
    """Update one reference using observer state and continuous weights."""

    def __init__(self, config: ApproachProfileConfig, k1: float) -> None:
        config.validate()
        if not math.isfinite(k1) or k1 <= 0.0:
            raise ValueError('k1 must be finite and positive')
        self.config = config
        self.k1 = k1
        self.segment: Optional[QuinticApproachSegment] = None
        self.target_speed_m_s = math.nan
        self.weight = math.nan
        self.replan_count = 0

    def reset(self) -> None:
        """Discard the reference when the mission resets its observer."""
        self.segment = None
        self.target_speed_m_s = math.nan
        self.weight = math.nan
        self.replan_count = 0

    def update(
        self,
        time_s: float,
        p_r_e: Sequence[float],
        v_r_e: Sequence[float],
        z1: float,
    ) -> ApproachReference:
        """Replan periodically and sample the reference at outer-loop time."""
        p_r = np.asarray(p_r_e, dtype=float)
        v_r = np.asarray(v_r_e, dtype=float)
        if (
            p_r.shape != (3,) or v_r.shape != (3,)
            or not np.all(np.isfinite(p_r))
            or not np.all(np.isfinite(v_r))
            or not math.isfinite(time_s) or not math.isfinite(z1)
            or z1 < 0.0
        ):
            raise ValueError(
                'approach state and time must be finite, z1 nonnegative'
            )
        distance = float(np.linalg.norm(p_r))
        # Keep the existing controller's numerical division domain.
        if distance <= 1e-6:
            raise ValueError('relative distance is too small for the controller')
        segment = self.segment
        if segment is not None and time_s < segment.start_time_s:
            self.reset()
            segment = None
        if (
            segment is None
            or time_s - segment.start_time_s >= 1.0 / self.config.replan_hz
            or time_s - segment.start_time_s >= segment.horizon_s
        ):
            n = -p_r / distance
            closing_speed = float(n @ v_r)
            transverse_velocity = v_r - closing_speed * n
            config = self.config
            nominal_speed = (
                self.k1 * distance
                + config.delta_v_m_s * distance
                / (distance + config.boost_length_m)
                + config.terminal_speed_m_s
                * math.exp(-distance / config.terminal_length_m)
            )
            weight = math.exp(
                -config.alpha * z1
                - config.beta
                * float(transverse_velocity @ transverse_velocity)
            )
            target_speed = weight * nominal_speed
            # Start with the complete old velocity reference, including its
            # derivative, so enabling planning leaves the first command intact.
            start = (
                segment.sample(time_s) if segment is not None
                else ApproachReference(
                    distance, self.k1 * distance, -self.k1 * closing_speed
                )
            )
            h = config.horizon_s
            # Geometric time scaling (proposal 5.1): all six Bernstein distance
            # coefficients remain positive with inherited acceleration.
            h = h * start.distance_m / (
                start.distance_m
                + h * (abs(start.speed_m_s) + target_speed) / 2.0
                + h**2 * abs(start.acceleration_m_s2) / 20.0
            )
            end = ApproachReference(
                start.distance_m - h * (start.speed_m_s + target_speed) / 2.0,
                target_speed, 0.0,
            )
            self.segment = QuinticApproachSegment.from_boundaries(
                time_s, h, start, end
            )
            self.target_speed_m_s = target_speed
            self.weight = weight
            self.replan_count += 1
        return self.segment.sample(time_s)
