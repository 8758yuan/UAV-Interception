"""Power-law mapping between physical and PX4 normalized thrust."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class ThrustMappingConfig:
    """Map physical thrust to the normalized PX4/Gazebo motor command."""

    mass_kg: float
    hover_thrust_normalized: float
    gravity_m_s2: float = 9.80665
    maximum_thrust_n: Optional[float] = None
    thrust_curve_exponent: float = 1.0
    actuator_minimum_fraction: float = 0.0

    def validate(self) -> None:
        """Reject values that cannot define a physical mapping."""
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError('mass_kg must be finite and positive')
        if (
            not math.isfinite(self.hover_thrust_normalized)
            or not 0.0 < self.hover_thrust_normalized <= 1.0
        ):
            raise ValueError(
                'hover_thrust_normalized must be within (0, 1]'
            )
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError('gravity_m_s2 must be finite and positive')
        if self.maximum_thrust_n is not None and (
            not math.isfinite(self.maximum_thrust_n)
            or self.maximum_thrust_n <= 0.0
        ):
            raise ValueError(
                'maximum_thrust_n must be finite and positive when provided'
            )
        if (
            not math.isfinite(self.thrust_curve_exponent)
            or self.thrust_curve_exponent <= 0.0
        ):
            raise ValueError('thrust_curve_exponent must be positive')
        if (
            not math.isfinite(self.actuator_minimum_fraction)
            or not 0.0 <= self.actuator_minimum_fraction < 1.0
        ):
            raise ValueError(
                'actuator_minimum_fraction must be within [0, 1)'
            )

    @property
    def thrust_at_full_command_n(self) -> float:
        """Return the calibrated thrust at a normalized command of one."""
        self.validate()
        if self.maximum_thrust_n is not None:
            return self.maximum_thrust_n
        rotor_fraction = (
            self.actuator_minimum_fraction
            + (1.0 - self.actuator_minimum_fraction)
            * self.hover_thrust_normalized
        )
        return (
            self.mass_kg
            * self.gravity_m_s2
            / rotor_fraction ** self.thrust_curve_exponent
        )


@dataclass(frozen=True)
class NormalizedThrust:
    """A bounded PX4 thrust magnitude and its saturation state."""

    magnitude: float
    body_z: float
    saturated: bool


def newtons_to_px4_normalized(
    thrust_n: float,
    config: ThrustMappingConfig,
) -> NormalizedThrust:
    """Map positive FLU +Z thrust to PX4 FRD negative body-Z command."""
    config.validate()
    if not math.isfinite(thrust_n) or thrust_n < 0.0:
        raise ValueError('thrust_n must be finite and nonnegative')
    thrust_fraction = thrust_n / config.thrust_at_full_command_n
    rotor_fraction = thrust_fraction ** (
        1.0 / config.thrust_curve_exponent
    )
    raw = (
        rotor_fraction - config.actuator_minimum_fraction
    ) / (1.0 - config.actuator_minimum_fraction)
    magnitude = min(max(raw, 0.0), 1.0)
    return NormalizedThrust(
        magnitude=magnitude,
        body_z=-magnitude,
        saturated=raw < 0.0 or raw > 1.0,
    )


def px4_normalized_to_newtons(
    magnitude: float,
    config: ThrustMappingConfig,
) -> float:
    """Convert a bounded PX4 thrust magnitude back to physical thrust."""
    config.validate()
    if not math.isfinite(magnitude) or not 0.0 <= magnitude <= 1.0:
        raise ValueError('magnitude must be within [0, 1]')
    rotor_fraction = (
        config.actuator_minimum_fraction
        + (1.0 - config.actuator_minimum_fraction) * magnitude
    )
    return rotor_fraction ** config.thrust_curve_exponent * (
        config.thrust_at_full_command_n
    )
