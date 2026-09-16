"""Pure conversion from controller FLU commands to PX4 FRD setpoints."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

from ibvs_control.thrust_mapping import (
    ThrustMappingConfig,
    newtons_to_px4_normalized,
)


@dataclass(frozen=True)
class Px4RateThrustCommand:
    """A bounded command in the PX4 body-FRD convention."""

    rates_frd_rad_s: Tuple[float, float, float]
    thrust_body: Tuple[float, float, float]
    rate_saturated: bool
    thrust_saturated: bool


def adapt_rate_thrust_command(
    omega_b_flu_rad_s: Sequence[float],
    thrust_n: float,
    omega_limit_rad_s: float,
    thrust_mapping: ThrustMappingConfig,
) -> Px4RateThrustCommand:
    """Apply a final safety bound and convert FLU axes/signs to PX4 FRD."""
    omega_flu = np.asarray(omega_b_flu_rad_s, dtype=float)
    if omega_flu.shape != (3,) or not np.all(np.isfinite(omega_flu)):
        raise ValueError('omega_b_flu_rad_s must be a finite three-vector')
    if not math.isfinite(omega_limit_rad_s) or omega_limit_rad_s <= 0.0:
        raise ValueError('omega_limit_rad_s must be finite and positive')

    omega_norm = float(np.linalg.norm(omega_flu))
    rate_saturated = omega_norm > omega_limit_rad_s
    if rate_saturated:
        omega_flu = omega_flu * (omega_limit_rad_s / omega_norm)

    # FLU -> FRD is a pi rotation about body X: [x, y, z] -> [x, -y, -z].
    omega_frd = np.array((omega_flu[0], -omega_flu[1], -omega_flu[2]))
    normalized = newtons_to_px4_normalized(thrust_n, thrust_mapping)
    return Px4RateThrustCommand(
        rates_frd_rad_s=tuple(float(value) for value in omega_frd),
        thrust_body=(0.0, 0.0, normalized.body_z),
        rate_saturated=rate_saturated,
        thrust_saturated=normalized.saturated,
    )
