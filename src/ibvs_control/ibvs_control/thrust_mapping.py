"""First-order mapping between physical and PX4 normalized thrust."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThrustMappingConfig:
    """Linear mapping inferred from a measured steady-hover command."""

    mass_kg: float
    hover_thrust_normalized: float
    gravity_m_s2: float = 9.80665

    def validate(self) -> None:
        """Reject values that cannot define a physical linear mapping."""
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

    @property
    def thrust_at_full_command_n(self) -> float:
        """Return the full-command thrust implied by the hover calibration."""
        self.validate()
        return (
            self.mass_kg
            * self.gravity_m_s2
            / self.hover_thrust_normalized
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
    raw = thrust_n / config.thrust_at_full_command_n
    magnitude = min(raw, 1.0)
    return NormalizedThrust(
        magnitude=magnitude,
        body_z=-magnitude,
        saturated=raw > 1.0,
    )


def px4_normalized_to_newtons(
    magnitude: float,
    config: ThrustMappingConfig,
) -> float:
    """Convert a bounded PX4 thrust magnitude back to physical thrust."""
    config.validate()
    if not math.isfinite(magnitude) or not 0.0 <= magnitude <= 1.0:
        raise ValueError('magnitude must be within [0, 1]')
    return magnitude * config.thrust_at_full_command_n
