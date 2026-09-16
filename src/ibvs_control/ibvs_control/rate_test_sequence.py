"""Pure command and response evaluator for the P1 body-rate direction test."""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


RateVector = Tuple[float, float, float]


@dataclass(frozen=True)
class RateTestConfig:
    """Configuration for bounded body-rate pulses with hover recovery."""

    roll_rate_rad_s: float = 0.30
    pitch_rate_rad_s: float = 0.30
    yaw_rate_rad_s: float = 0.45
    prepare_duration_s: float = 2.0
    pulse_duration_s: float = 0.4
    settle_duration_s: float = 2.0
    response_ignore_s: float = 0.15
    minimum_response_rad_s: float = 0.02
    minimum_samples: int = 3

    def validate(self) -> None:
        """Reject settings that cannot produce a meaningful safe test."""
        rates = (
            self.roll_rate_rad_s,
            self.pitch_rate_rad_s,
            self.yaw_rate_rad_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in rates):
            raise ValueError('body rates must be finite and greater than zero')
        if any(value > 0.5 for value in rates):
            raise ValueError('P1 body rates must not exceed 0.5 rad/s')

        durations = (
            self.prepare_duration_s,
            self.pulse_duration_s,
            self.settle_duration_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in durations):
            raise ValueError('test durations must be finite and positive')
        if (
            not math.isfinite(self.response_ignore_s)
            or self.response_ignore_s < 0.0
            or self.response_ignore_s >= self.pulse_duration_s
        ):
            raise ValueError(
                'response_ignore_s must be within the pulse duration'
            )
        if (
            not math.isfinite(self.minimum_response_rad_s)
            or self.minimum_response_rad_s <= 0.0
        ):
            raise ValueError('minimum response must be finite and positive')
        if self.minimum_samples <= 0:
            raise ValueError('minimum_samples must be greater than zero')


@dataclass(frozen=True)
class RateTestResult:
    """Measured response for one signed command pulse."""

    name: str
    commanded_rate_rad_s: float
    mean_rate_rad_s: float
    samples: int
    passed: bool
    reason: str


@dataclass(frozen=True)
class _Segment:
    name: str
    duration_s: float
    command: RateVector
    measured_axis: Optional[int] = None


class RateTestSequence:
    """Generate signed FRD rate pulses and verify measured response signs."""

    def __init__(self, config: RateTestConfig) -> None:
        config.validate()
        self.config = config
        self._segments = self._build_segments(config)
        self._index = 0
        self._segment_started_s: Optional[float] = None
        self._samples: List[float] = []
        self.results: List[RateTestResult] = []
        self._pending_results: List[RateTestResult] = []

    @property
    def complete(self) -> bool:
        """Return whether all six signed axis pulses have completed."""
        return self._index >= len(self._segments)

    @property
    def passed(self) -> bool:
        """Return true only if the complete sequence passed every pulse."""
        return self.complete and bool(self.results) and all(
            result.passed for result in self.results
        )

    @property
    def current_name(self) -> str:
        """Return the active segment name for logging."""
        if self.complete:
            return 'COMPLETE'
        return self._segments[self._index].name

    @property
    def uses_position_control(self) -> bool:
        """Return whether the active segment should recover in hover mode."""
        if self.complete:
            return True
        return self._segments[self._index].measured_axis is None

    def command(
        self,
        now_s: float,
        recovery_ready: bool = True,
    ) -> RateVector:
        """Return the command, optionally holding recovery until stabilized."""
        if self._segment_started_s is None:
            self._segment_started_s = now_s

        while not self.complete:
            segment = self._segments[self._index]
            elapsed_s = now_s - self._segment_started_s
            if elapsed_s < segment.duration_s:
                return segment.command
            gated_recovery = (
                segment.measured_axis is None
                and segment.name.endswith('_RECOVER')
            )
            if gated_recovery and not recovery_ready:
                return segment.command
            self._finalize_segment(segment)
            self._index += 1
            if gated_recovery:
                self._segment_started_s = now_s
            else:
                self._segment_started_s += segment.duration_s
            self._samples = []

        return 0.0, 0.0, 0.0

    def observe(self, xyz_rad_s: Sequence[float], now_s: float) -> None:
        """Record the active-axis response after its transient ignore period."""
        if self.complete or self._segment_started_s is None:
            return
        if len(xyz_rad_s) != 3:
            raise ValueError('angular velocity must contain three values')
        measured = tuple(float(value) for value in xyz_rad_s)
        if not all(math.isfinite(value) for value in measured):
            raise ValueError('angular velocity values must be finite')

        segment = self._segments[self._index]
        if segment.measured_axis is None:
            return
        if now_s - self._segment_started_s < self.config.response_ignore_s:
            return
        self._samples.append(measured[segment.measured_axis])

    def pop_results(self) -> Tuple[RateTestResult, ...]:
        """Return newly completed pulse results exactly once."""
        results = tuple(self._pending_results)
        self._pending_results.clear()
        return results

    def _finalize_segment(self, segment: _Segment) -> None:
        if segment.measured_axis is None:
            return

        commanded_rate = segment.command[segment.measured_axis]
        sample_count = len(self._samples)
        mean_rate = (
            sum(self._samples) / sample_count if sample_count else 0.0
        )
        if sample_count < self.config.minimum_samples:
            passed = False
            reason = 'insufficient_samples'
        elif abs(mean_rate) < self.config.minimum_response_rad_s:
            passed = False
            reason = 'response_too_small'
        elif mean_rate * commanded_rate <= 0.0:
            passed = False
            reason = 'wrong_sign'
        else:
            passed = True
            reason = 'ok'

        result = RateTestResult(
            name=segment.name,
            commanded_rate_rad_s=commanded_rate,
            mean_rate_rad_s=mean_rate,
            samples=sample_count,
            passed=passed,
            reason=reason,
        )
        self.results.append(result)
        self._pending_results.append(result)

    @staticmethod
    def _build_segments(config: RateTestConfig) -> Tuple[_Segment, ...]:
        zero = (0.0, 0.0, 0.0)
        pulse = config.pulse_duration_s
        settle = config.settle_duration_s
        segments = [_Segment('PREPARE', config.prepare_duration_s, zero)]
        axes = (
            ('ROLL', 0, config.roll_rate_rad_s),
            ('PITCH', 1, config.pitch_rate_rad_s),
            ('YAW', 2, config.yaw_rate_rad_s),
        )
        for name, axis, magnitude in axes:
            positive = [0.0, 0.0, 0.0]
            positive[axis] = magnitude
            negative = [0.0, 0.0, 0.0]
            negative[axis] = -magnitude
            segments.extend((
                _Segment(
                    f'{name}_POSITIVE',
                    pulse,
                    tuple(positive),
                    axis,
                ),
                _Segment(f'{name}_POSITIVE_RECOVER', settle, zero),
                _Segment(
                    f'{name}_NEGATIVE',
                    pulse,
                    tuple(negative),
                    axis,
                ),
                _Segment(f'{name}_NEGATIVE_RECOVER', settle, zero),
            ))
        return tuple(segments)
