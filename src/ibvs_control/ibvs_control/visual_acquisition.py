"""Target-state-free visual search and alignment before interception."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class VisualAcquisitionPhase(str, Enum):
    """Camera-only phases used before forward interception is authorized."""

    SEARCH = 'VISUAL_SEARCH'
    ALIGN = 'VISUAL_ALIGN'
    READY = 'VISUAL_READY'


@dataclass(frozen=True)
class VisualAcquisitionConfig:
    """Limits for position-hold search and pixel-error alignment."""

    search_yaw_rate_rad_s: float
    align_yaw_gain_rad_s: float
    align_vertical_gain_m_s: float
    center_error: float
    release_error: float
    settle_time_s: float
    target_loss_timeout_s: float
    maximum_vertical_offset_m: float

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent acquisition tuning."""
        values = {
            'search_yaw_rate_rad_s': self.search_yaw_rate_rad_s,
            'align_yaw_gain_rad_s': self.align_yaw_gain_rad_s,
            'align_vertical_gain_m_s': self.align_vertical_gain_m_s,
            'center_error': self.center_error,
            'release_error': self.release_error,
            'settle_time_s': self.settle_time_s,
            'target_loss_timeout_s': self.target_loss_timeout_s,
            'maximum_vertical_offset_m': self.maximum_vertical_offset_m,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.release_error <= self.center_error:
            raise ValueError('release_error must exceed center_error')


@dataclass(frozen=True)
class VisualAcquisitionCommand:
    """Relative hover adjustments produced only from image features."""

    phase: VisualAcquisitionPhase
    yaw_setpoint_rad: float
    vertical_offset_m: float
    image_error: float
    ready: bool


class VisualAcquisition:
    """Search for, center, and dwell on a camera target while hovering."""

    def __init__(self, config: VisualAcquisitionConfig) -> None:
        config.validate()
        self.config = config
        self.phase = VisualAcquisitionPhase.SEARCH
        self.yaw_setpoint_rad: Optional[float] = None
        self.vertical_offset_m = 0.0
        self.last_step_s: Optional[float] = None
        self.last_seen_s: Optional[float] = None
        self.centered_since_s: Optional[float] = None

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
        elapsed_s = 0.0
        if self.last_step_s is not None:
            elapsed_s = max(0.0, now_s - self.last_step_s)
        self.last_step_s = now_s

        image_error = math.hypot(x_norm, y_norm) if target_detected else math.inf
        if target_detected:
            self.last_seen_s = now_s

        if self.phase == VisualAcquisitionPhase.SEARCH:
            self.centered_since_s = None
            if target_detected:
                self.phase = VisualAcquisitionPhase.ALIGN
                self.yaw_setpoint_rad = _wrap_angle(current_yaw_rad)
            else:
                self.yaw_setpoint_rad = _wrap_angle(
                    self.yaw_setpoint_rad
                    + self.config.search_yaw_rate_rad_s * elapsed_s
                )

        if self.phase == VisualAcquisitionPhase.ALIGN:
            if target_detected:
                self.yaw_setpoint_rad = _wrap_angle(
                    self.yaw_setpoint_rad
                    + self.config.align_yaw_gain_rad_s * x_norm * elapsed_s
                )
                self.vertical_offset_m = _clip(
                    self.vertical_offset_m
                    + self.config.align_vertical_gain_m_s
                    * y_norm
                    * elapsed_s,
                    -self.config.maximum_vertical_offset_m,
                    self.config.maximum_vertical_offset_m,
                )
                if image_error <= self.config.center_error:
                    if self.centered_since_s is None:
                        self.centered_since_s = now_s
                    elif now_s - self.centered_since_s >= self.config.settle_time_s:
                        self.phase = VisualAcquisitionPhase.READY
                elif image_error >= self.config.release_error:
                    self.centered_since_s = None
            elif (
                self.last_seen_s is None
                or now_s - self.last_seen_s
                >= self.config.target_loss_timeout_s
            ):
                self.phase = VisualAcquisitionPhase.SEARCH
                self.centered_since_s = None
                self.yaw_setpoint_rad = _wrap_angle(current_yaw_rad)

        return VisualAcquisitionCommand(
            phase=self.phase,
            yaw_setpoint_rad=self.yaw_setpoint_rad,
            vertical_offset_m=self.vertical_offset_m,
            image_error=image_error,
            ready=self.phase == VisualAcquisitionPhase.READY,
        )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
