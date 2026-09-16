"""Pure accumulation of controller-shadow acceptance metrics."""

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional


@dataclass(frozen=True)
class ShadowSummary:
    """Serializable metrics for one read-only controller observation."""

    sample_count: int
    valid_count: int
    duration_s: float
    sample_rate_hz: float
    valid_fraction: float
    min_barrier_margin: Optional[float]
    max_abs_z1: Optional[float]
    max_thrust_n: Optional[float]
    max_thrust_normalized: Optional[float]
    thrust_saturation_fraction: float
    thrust_mapping_saturation_fraction: float
    rate_saturation_fraction: float
    invalid_reasons: Dict[str, int]

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""
        return asdict(self)


class ShadowMetrics:
    """Accumulate debug samples without depending on ROS message classes."""

    def __init__(self, k_b: float) -> None:
        if not math.isfinite(k_b) or k_b <= 0.0:
            raise ValueError('k_b must be finite and positive')
        self.k_b = k_b
        self.sample_count = 0
        self.valid_count = 0
        self.first_stamp_s: Optional[float] = None
        self.last_stamp_s: Optional[float] = None
        self.min_barrier_margin: Optional[float] = None
        self.max_abs_z1: Optional[float] = None
        self.max_thrust_n: Optional[float] = None
        self.max_thrust_normalized: Optional[float] = None
        self.thrust_saturation_count = 0
        self.thrust_mapping_saturation_count = 0
        self.rate_saturation_count = 0
        self.invalid_reasons: Counter = Counter()

    def update(
        self,
        *,
        stamp_s: float,
        valid: bool,
        reason: str,
        z1: float = 0.0,
        thrust_n: float = 0.0,
        thrust_normalized: float = 0.0,
        thrust_saturated: bool = False,
        thrust_mapping_saturated: bool = False,
        rate_saturated: bool = False,
    ) -> None:
        """Consume one debug sample and update all aggregate statistics."""
        if not math.isfinite(stamp_s):
            raise ValueError('stamp_s must be finite')
        if self.last_stamp_s is not None and stamp_s < self.last_stamp_s:
            raise ValueError('sample timestamps must be monotonic')
        if self.first_stamp_s is None:
            self.first_stamp_s = stamp_s
        self.last_stamp_s = stamp_s
        self.sample_count += 1

        if not valid:
            self.invalid_reasons[reason or 'unspecified'] += 1
            return
        values = (z1, thrust_n, thrust_normalized)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('valid sample metrics must be finite')
        self.valid_count += 1
        abs_z1 = abs(z1)
        margin = self.k_b - abs_z1
        self.min_barrier_margin = _minimum(self.min_barrier_margin, margin)
        self.max_abs_z1 = _maximum(self.max_abs_z1, abs_z1)
        self.max_thrust_n = _maximum(self.max_thrust_n, thrust_n)
        self.max_thrust_normalized = _maximum(
            self.max_thrust_normalized,
            thrust_normalized,
        )
        self.thrust_saturation_count += int(thrust_saturated)
        self.thrust_mapping_saturation_count += int(
            thrust_mapping_saturated
        )
        self.rate_saturation_count += int(rate_saturated)

    @property
    def duration_s(self) -> float:
        """Return elapsed message time covered by the observation."""
        if self.first_stamp_s is None or self.last_stamp_s is None:
            return 0.0
        return self.last_stamp_s - self.first_stamp_s

    def summary(self) -> ShadowSummary:
        """Freeze current statistics into a serializable value object."""
        duration = self.duration_s
        sample_rate = (
            (self.sample_count - 1) / duration
            if self.sample_count > 1 and duration > 0.0
            else 0.0
        )
        valid_denominator = max(self.valid_count, 1)
        return ShadowSummary(
            sample_count=self.sample_count,
            valid_count=self.valid_count,
            duration_s=duration,
            sample_rate_hz=sample_rate,
            valid_fraction=(
                self.valid_count / self.sample_count
                if self.sample_count
                else 0.0
            ),
            min_barrier_margin=self.min_barrier_margin,
            max_abs_z1=self.max_abs_z1,
            max_thrust_n=self.max_thrust_n,
            max_thrust_normalized=self.max_thrust_normalized,
            thrust_saturation_fraction=(
                self.thrust_saturation_count / valid_denominator
            ),
            thrust_mapping_saturation_fraction=(
                self.thrust_mapping_saturation_count / valid_denominator
            ),
            rate_saturation_fraction=(
                self.rate_saturation_count / valid_denominator
            ),
            invalid_reasons=dict(self.invalid_reasons),
        )


def _minimum(current: Optional[float], value: float) -> float:
    return value if current is None else min(current, value)


def _maximum(current: Optional[float], value: float) -> float:
    return value if current is None else max(current, value)
