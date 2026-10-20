"""Target-state-free visual search and detection before interception."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class VisualAcquisitionPhase(str, Enum):
    """Camera-only search and target-visibility phases."""

    SEARCH = 'VISUAL_SEARCH'
    READY = 'VISUAL_READY'


@dataclass(frozen=True)
class VisualAcquisitionConfig:
    """
    Parameters for position-hold search and target visibility.

    Search is deliberately active in both yaw and altitude.  The vehicle
    searches for targets at a different elevation without receiving target
    world-state information. Detection stops the scan without alignment.
    """

    search_yaw_rate_rad_s: float
    search_vertical_amplitude_m: float
    search_vertical_period_s: float
    desired_image_xy: tuple[float, float]
    target_loss_timeout_s: float

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent acquisition tuning."""
        values = {
            'search_yaw_rate_rad_s': self.search_yaw_rate_rad_s,
            'search_vertical_amplitude_m': self.search_vertical_amplitude_m,
            'search_vertical_period_s': self.search_vertical_period_s,
            'target_loss_timeout_s': self.target_loss_timeout_s,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if (
            len(self.desired_image_xy) != 2
            or not all(math.isfinite(v) for v in self.desired_image_xy)
        ):
            raise ValueError('desired_image_xy must contain two finite values')


@dataclass(frozen=True)
class VisualAcquisitionCommand:
    """Search setpoints and current target availability."""

    phase: VisualAcquisitionPhase
    yaw_setpoint_rad: float
    vertical_offset_m: float
    image_error: float
    ready: bool


class VisualAcquisition:
    """Search until detection, then report readiness without alignment."""

    def __init__(self, config: VisualAcquisitionConfig) -> None:
        config.validate()
        self.config = config
        self.phase = VisualAcquisitionPhase.SEARCH
        self.yaw_setpoint_rad: Optional[float] = None
        self.vertical_offset_m = 0.0
        self.last_step_s: Optional[float] = None
        self.last_seen_s: Optional[float] = None
        self.search_started_s: Optional[float] = None
        self.search_center_offset_m = 0.0

    def step(
        self,
        *,
        now_s: float,
        current_yaw_rad: float,
        target_detected: bool,
        x_norm: float = 0.0,
        y_norm: float = 0.0,
    ) -> VisualAcquisitionCommand:
        """Advance acquisition using pixels and the interceptor's own yaw."""
        values = (now_s, current_yaw_rad, x_norm, y_norm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('visual acquisition inputs must be finite')
        if now_s < 0.0:
            raise ValueError('now_s must be nonnegative')
        if self.yaw_setpoint_rad is None:
            self.yaw_setpoint_rad = _wrap_angle(current_yaw_rad)
        if self.search_started_s is None:
            self.search_started_s = now_s
        elapsed_s = 0.0
        if self.last_step_s is not None:
            elapsed_s = max(0.0, now_s - self.last_step_s)
        self.last_step_s = now_s

        desired_x, desired_y = self.config.desired_image_xy
        x_error = x_norm - desired_x
        y_error = y_norm - desired_y
        image_error = (
            math.hypot(x_error, y_error) if target_detected else math.inf
        )
        if target_detected:
            self.last_seen_s = now_s
            if self.phase == VisualAcquisitionPhase.SEARCH:
                self.yaw_setpoint_rad = _wrap_angle(current_yaw_rad)
            self.phase = VisualAcquisitionPhase.READY
        elif self.phase == VisualAcquisitionPhase.READY:
            # A dropout removes readiness immediately; the existing loss
            # timeout prevents a single dropped frame from restarting search.
            if (
                self.last_seen_s is None
                or now_s - self.last_seen_s
                >= self.config.target_loss_timeout_s
            ):
                self._begin_search(now_s, current_yaw_rad)

        elif self.phase == VisualAcquisitionPhase.SEARCH:
            self.yaw_setpoint_rad = _wrap_angle(
                self.yaw_setpoint_rad
                + self.config.search_yaw_rate_rad_s * elapsed_s
            )
            self.vertical_offset_m = self._search_vertical_offset(now_s)

        return VisualAcquisitionCommand(
            phase=self.phase,
            yaw_setpoint_rad=self.yaw_setpoint_rad,
            vertical_offset_m=self.vertical_offset_m,
            image_error=image_error,
            ready=target_detected,
        )

    def _begin_search(self, now_s: float, current_yaw_rad: float) -> None:
        """Start a fresh bounded yaw-and-height search around current pose."""
        self.phase = VisualAcquisitionPhase.SEARCH
        self.yaw_setpoint_rad = _wrap_angle(current_yaw_rad)
        self.search_started_s = now_s
        self.search_center_offset_m = self.vertical_offset_m

    def _search_vertical_offset(self, now_s: float) -> float:
        """Return a smooth vertical scan offset centered on the last pose."""
        if self.search_started_s is None:
            return self.vertical_offset_m
        elapsed_s = max(0.0, now_s - self.search_started_s)
        phase = (
            2.0 * math.pi * elapsed_s
            / self.config.search_vertical_period_s
        )
        offset = (
            self.search_center_offset_m
            + self.config.search_vertical_amplitude_m * math.sin(phase)
        )
        return offset


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
