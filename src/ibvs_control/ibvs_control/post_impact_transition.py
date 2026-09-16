"""Continuous velocity profile for leaving interception after impact."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SmoothStopCommand:
    """Velocity and matching acceleration along a smooth stop profile."""

    velocity_ned: np.ndarray
    acceleration_ned: np.ndarray
    progress: float


def smooth_stop_command(
    initial_velocity_ned: Sequence[float],
    elapsed_s: float,
    duration_s: float,
) -> SmoothStopCommand:
    """Decay velocity to zero with zero acceleration at both endpoints."""
    velocity = np.asarray(initial_velocity_ned, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError('initial_velocity_ned must be a finite three-vector')
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError('elapsed_s must be finite and nonnegative')
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError('duration_s must be finite and positive')

    progress = min(elapsed_s / duration_s, 1.0)
    smoothstep = progress * progress * (3.0 - 2.0 * progress)
    scale = 1.0 - smoothstep
    scale_rate = -6.0 * progress * (1.0 - progress) / duration_s
    return SmoothStopCommand(
        velocity_ned=scale * velocity,
        acceleration_ned=scale_rate * velocity,
        progress=progress,
    )
