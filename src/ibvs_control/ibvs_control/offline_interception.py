"""Deterministic offline closed-loop simulation for P2 controller tuning."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

from ibvs_control.so3_controller import (
    OuterLoopConfig,
    compute_inner_loop,
    compute_outer_loop,
    rodrigues,
)


@dataclass(frozen=True)
class OfflineSimulationConfig:
    """Timing, initial-state, and safety settings for a P2 offline run."""

    outer_loop_hz: int = 50
    inner_loop_hz: int = 200
    duration_s: float = 40.0
    hit_radius_m: float = 0.5
    speed_limit_m_s: float = 2.0
    tilt_limit_deg: float = 20.0
    omega_max_rad_s: float = 0.5
    initial_p_r_e: tuple = (-12.0, -1.0, -1.0)
    initial_v_r_e: tuple = (0.0, 0.0, 0.0)
    camera_axis_b: tuple = (1.0, 0.0, 0.0)
    designed_los_pitch_deg: float = 30.0

    def validate(self) -> None:
        """Reject timing and safety settings unsuitable for integration."""
        if self.outer_loop_hz <= 0 or self.inner_loop_hz <= 0:
            raise ValueError('loop frequencies must be positive')
        if self.inner_loop_hz % self.outer_loop_hz != 0:
            raise ValueError('inner_loop_hz must be divisible by outer_loop_hz')
        positive = (
            self.duration_s,
            self.hit_radius_m,
            self.speed_limit_m_s,
            self.tilt_limit_deg,
            self.omega_max_rad_s,
            self.designed_los_pitch_deg,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('duration and safety limits must be positive')


@dataclass(frozen=True)
class OfflineSimulationResult:
    """Metrics produced by one deterministic offline interception run."""

    hit: bool
    safe: bool
    reason: str
    elapsed_s: float
    d_min_m: float
    max_speed_m_s: float
    max_tilt_deg: float
    max_los_error_deg: float
    rate_saturation_fraction: float
    thrust_saturation_fraction: float
    final_p_r_e: tuple


def simulate_interception(
    controller: OuterLoopConfig,
    simulation: OfflineSimulationConfig = OfflineSimulationConfig(),
) -> OfflineSimulationResult:
    """Simulate the paper's 50 Hz outer and 200 Hz attitude loops."""
    controller.validate()
    simulation.validate()
    p_r = _vector3(simulation.initial_p_r_e, 'initial_p_r_e')
    v_r = _vector3(simulation.initial_v_r_e, 'initial_v_r_e')
    camera_axis_b = _unit3(simulation.camera_axis_b, 'camera_axis_b')
    designed_pitch = math.radians(simulation.designed_los_pitch_deg)
    designed_los_b = np.array(
        (math.cos(designed_pitch), 0.0, math.sin(designed_pitch))
    )
    attitude = np.eye(3)
    gravity = np.array((0.0, 0.0, -controller.gravity_m_s2))
    dt_s = 1.0 / simulation.inner_loop_hz
    outer_period_steps = simulation.inner_loop_hz // simulation.outer_loop_hz
    total_steps = int(round(simulation.duration_s * simulation.inner_loop_hz))

    d_min_m = float(np.linalg.norm(p_r))
    max_speed_m_s = float(np.linalg.norm(v_r))
    max_tilt_deg = 0.0
    max_los_error_deg = 0.0
    rate_saturated_steps = 0
    thrust_saturated_steps = 0
    outer = None
    reason = 'timeout'
    hit = False
    step = 0

    for step in range(total_steps + 1):
        distance_m = float(np.linalg.norm(p_r))
        d_min_m = min(d_min_m, distance_m)
        if distance_m <= simulation.hit_radius_m:
            hit = True
            reason = 'hit'
            break

        los_e = -p_r / distance_m
        if step % outer_period_steps == 0:
            designed_los_e = attitude @ designed_los_b
            try:
                outer = compute_outer_loop(
                    p_r,
                    v_r,
                    los_e,
                    designed_los_e,
                    attitude,
                    controller,
                )
            except ValueError as error:
                reason = f'controller_error:{error}'
                break
        if outer is None:
            raise RuntimeError('outer-loop state was not initialized')

        inner = compute_inner_loop(
            outer,
            attitude,
            controller,
            simulation.omega_max_rad_s,
        )
        omega_b = inner.omega_d_b
        if inner.rate_saturated:
            rate_saturated_steps += 1
        if inner.thrust_saturated:
            thrust_saturated_steps += 1

        rate_norm = float(np.linalg.norm(omega_b))
        if rate_norm > 1e-12:
            attitude = attitude @ rodrigues(
                omega_b / rate_norm,
                rate_norm * dt_s,
            )
        acceleration_e = gravity + (
            inner.thrust_n / controller.mass_kg * attitude[:, 2]
        )
        v_r = v_r + acceleration_e * dt_s
        p_r = p_r + v_r * dt_s

        speed_m_s = float(np.linalg.norm(v_r))
        tilt_cosine = float(np.clip(attitude[2, 2], -1.0, 1.0))
        los_cosine = float(np.clip((attitude @ camera_axis_b) @ los_e, -1.0, 1.0))
        max_speed_m_s = max(max_speed_m_s, speed_m_s)
        max_tilt_deg = max(
            max_tilt_deg,
            math.degrees(math.acos(tilt_cosine)),
        )
        max_los_error_deg = max(
            max_los_error_deg,
            math.degrees(math.acos(los_cosine)),
        )

    elapsed_s = min(step * dt_s, simulation.duration_s)
    samples = max(step, 1)
    safe = (
        hit
        and max_speed_m_s <= simulation.speed_limit_m_s
        and max_tilt_deg <= simulation.tilt_limit_deg
    )
    if hit and not safe:
        reason = 'hit_with_safety_limit_violation'
    return OfflineSimulationResult(
        hit=hit,
        safe=safe,
        reason=reason,
        elapsed_s=elapsed_s,
        d_min_m=d_min_m,
        max_speed_m_s=max_speed_m_s,
        max_tilt_deg=max_tilt_deg,
        max_los_error_deg=max_los_error_deg,
        rate_saturation_fraction=rate_saturated_steps / samples,
        thrust_saturation_fraction=thrust_saturated_steps / samples,
        final_p_r_e=tuple(float(value) for value in p_r),
    )


def default_controller_config() -> OuterLoopConfig:
    """Return the first offline-safe tuning candidate, not a paper parameter."""
    return OuterLoopConfig(
        k1=0.02,
        k2=20.0,
        k_b=1.0 - math.cos(math.radians(60.0)),
        mass_kg=2.0,
        thrust_max_n=26.9784,
    )


def main(args: Sequence[str] = None) -> int:
    """Run the default offline case and print a machine-readable report."""
    parser = argparse.ArgumentParser(description='Run the P2 offline intercept')
    parser.add_argument('--duration-s', type=float, default=40.0)
    options = parser.parse_args(args)
    simulation = OfflineSimulationConfig(duration_s=options.duration_s)
    result = simulate_interception(default_controller_config(), simulation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.safe else 1


def _vector3(values, name: str) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite three-vector')
    return value


def _unit3(values, name: str) -> np.ndarray:
    value = _vector3(values, name)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f'{name} must be nonzero')
    return value / norm


if __name__ == '__main__':
    raise SystemExit(main())
