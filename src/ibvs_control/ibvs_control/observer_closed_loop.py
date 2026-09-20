"""Deterministic no-delay static-target loop through the paper observer."""

from dataclasses import dataclass
import math

import numpy as np

from ibvs_control.frames import (
    quaternion_wxyz_to_rotation,
    rotation_to_quaternion_wxyz,
)
from ibvs_control.offline_interception import OfflineSimulationConfig
from ibvs_control.paper_state_observer import (
    ObserverConfig,
    ObserverNoise,
    PaperStateObserver,
)
from ibvs_control.so3_controller import (
    OuterLoopConfig,
    compute_inner_loop,
    compute_outer_loop,
    image_los_in_earth,
    rodrigues,
)


CAMERA_TO_BODY = np.array(
    ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
)


@dataclass(frozen=True)
class ObserverClosedLoopResult:
    """Acceptance metrics for the complete camera-observer-controller loop."""

    hit: bool
    safe: bool
    reason: str
    elapsed_s: float
    minimum_distance_m: float
    maximum_speed_m_s: float
    maximum_tilt_deg: float
    maximum_image_radius: float
    maximum_position_error_m: float
    maximum_velocity_error_m_s: float
    prediction_count: int
    correction_count: int
    target_distance_m: float


def simulate_observer_closed_loop(
    controller: OuterLoopConfig,
    simulation: OfflineSimulationConfig = OfflineSimulationConfig(),
    camera_hz: float = 30.0,
    observer_noise: ObserverNoise = ObserverNoise(),
    observer_config: ObserverConfig = ObserverConfig(),
    target_velocity_e: tuple = (0.0, 0.0, 0.0),
) -> ObserverClosedLoopResult:
    """
    Run a constant-velocity-target plant using observer estimates for control.

    Ground truth is confined to the plant and synthetic camera.  The control
    path receives the same normalized image, IMU, and initial attitude data as
    the ROS graph and never receives target position or target velocity.
    """
    controller.validate()
    simulation.validate()
    if not math.isfinite(camera_hz) or camera_hz <= 0.0:
        raise ValueError('camera_hz must be finite and positive')
    target_velocity = np.asarray(target_velocity_e, dtype=float)
    if target_velocity.shape != (3,) or not np.all(np.isfinite(target_velocity)):
        raise ValueError('target_velocity_e must contain three finite values')

    p_r_true = np.asarray(simulation.initial_p_r_e, dtype=float)
    vehicle_velocity_true = np.asarray(
        simulation.initial_v_r_e,
        dtype=float,
    )
    v_r_true = vehicle_velocity_true - target_velocity
    attitude_true = np.eye(3)
    image = _project_target(p_r_true, attitude_true)
    initial_depth = _target_in_camera(p_r_true, attitude_true)[2]
    observer = PaperStateObserver(
        config=observer_config,
        noise=observer_noise,
    )
    observer.initialize_from_image(
        rotation_to_quaternion_wxyz(attitude_true),
        image,
        float(initial_depth),
    )

    dt_s = 1.0 / simulation.inner_loop_hz
    outer_period_steps = simulation.inner_loop_hz // simulation.outer_loop_hz
    camera_period_s = 1.0 / camera_hz
    next_camera_s = camera_period_s
    total_steps = int(round(simulation.duration_s * simulation.inner_loop_hz))
    designed_pitch = math.radians(simulation.designed_los_pitch_deg)
    designed_los_b = np.array(
        (math.cos(designed_pitch), 0.0, math.sin(designed_pitch))
    )
    gravity = np.array((0.0, 0.0, -controller.gravity_m_s2))

    minimum_distance = float(np.linalg.norm(p_r_true))
    maximum_speed = 0.0
    maximum_tilt = 0.0
    maximum_image_radius = float(np.linalg.norm(image))
    maximum_position_error = 0.0
    maximum_velocity_error = 0.0
    outer = None
    hit = False
    reason = 'timeout'
    step = 0

    for step in range(total_steps + 1):
        time_s = step * dt_s
        distance = float(np.linalg.norm(p_r_true))
        minimum_distance = min(minimum_distance, distance)
        if distance <= simulation.hit_radius_m:
            hit = True
            reason = 'hit'
            break

        estimate = observer.snapshot().state
        attitude_estimate = quaternion_wxyz_to_rotation(estimate.q)
        if step % outer_period_steps == 0:
            los_estimate = image_los_in_earth(
                estimate.image_xy,
                attitude_estimate,
                CAMERA_TO_BODY,
            )
            try:
                outer = compute_outer_loop(
                    estimate.p_r_e,
                    estimate.v_r_e,
                    los_estimate,
                    attitude_estimate @ designed_los_b,
                    attitude_estimate,
                    controller,
                )
            except ValueError as error:
                reason = f'controller_error:{error}'
                break
        if outer is None:
            raise RuntimeError('outer loop did not initialize')
        inner = compute_inner_loop(
            outer,
            attitude_estimate,
            controller,
            simulation.omega_max_rad_s,
        )

        rate_norm = float(np.linalg.norm(inner.omega_d_b))
        if rate_norm > 1e-12:
            attitude_true = attitude_true @ rodrigues(
                inner.omega_d_b / rate_norm,
                rate_norm * dt_s,
            )
        acceleration_true = gravity + (
            inner.thrust_n / controller.mass_kg * attitude_true[:, 2]
        )
        vehicle_velocity_true = (
            vehicle_velocity_true + acceleration_true * dt_s
        )
        v_r_true = vehicle_velocity_true - target_velocity
        p_r_true = p_r_true + v_r_true * dt_s

        specific_force_b = attitude_true.T @ (acceleration_true - gravity)
        try:
            observer.predict(inner.omega_d_b, specific_force_b, dt_s)
        except ValueError as error:
            reason = f'observer_error:{error}'
            break
        if time_s + dt_s + 1e-12 >= next_camera_s:
            image = _project_target(p_r_true, attitude_true)
            observer.correct_image(image)
            next_camera_s += camera_period_s

        estimate = observer.snapshot().state
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(np.asarray(estimate.p_r_e) - p_r_true)),
        )
        maximum_velocity_error = max(
            maximum_velocity_error,
            float(np.linalg.norm(np.asarray(estimate.v_r_e) - v_r_true)),
        )
        maximum_speed = max(
            maximum_speed,
            float(np.linalg.norm(vehicle_velocity_true)),
        )
        maximum_tilt = max(
            maximum_tilt,
            math.degrees(math.acos(float(np.clip(attitude_true[2, 2], -1, 1)))),
        )
        maximum_image_radius = max(
            maximum_image_radius,
            float(np.linalg.norm(_project_target(p_r_true, attitude_true))),
        )

    snapshot = observer.snapshot()
    safe = bool(
        hit
        and maximum_speed <= simulation.speed_limit_m_s
        and maximum_tilt <= simulation.tilt_limit_deg
        and maximum_image_radius < math.tan(math.radians(39.0))
    )
    if hit and not safe:
        reason = 'hit_with_safety_limit_violation'
    return ObserverClosedLoopResult(
        hit=hit,
        safe=safe,
        reason=reason,
        elapsed_s=min(step * dt_s, simulation.duration_s),
        minimum_distance_m=minimum_distance,
        maximum_speed_m_s=maximum_speed,
        maximum_tilt_deg=maximum_tilt,
        maximum_image_radius=maximum_image_radius,
        maximum_position_error_m=maximum_position_error,
        maximum_velocity_error_m_s=maximum_velocity_error,
        prediction_count=snapshot.prediction_count,
        correction_count=snapshot.correction_count,
        target_distance_m=(
            float(np.linalg.norm(target_velocity))
            * min(step * dt_s, simulation.duration_s)
        ),
    )


def _target_in_camera(p_r_e: np.ndarray, attitude_b_to_e: np.ndarray) -> np.ndarray:
    target_from_interceptor_e = -np.asarray(p_r_e, dtype=float)
    return CAMERA_TO_BODY.T @ attitude_b_to_e.T @ target_from_interceptor_e


def _project_target(p_r_e: np.ndarray, attitude_b_to_e: np.ndarray) -> np.ndarray:
    target_c = _target_in_camera(p_r_e, attitude_b_to_e)
    if target_c[2] <= 0.1:
        raise ValueError('target crossed or left the forward camera plane')
    return target_c[:2] / target_c[2]
