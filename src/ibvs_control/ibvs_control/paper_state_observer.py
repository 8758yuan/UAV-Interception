"""18-state delayed Kalman observer from the paper's Appendix B and C."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


STATE_DIM = 18
PROCESS_NOISE_DIM = 6
MEASUREMENT_DIM = 2

Q_SLICE = slice(0, 4)
P_R_SLICE = slice(4, 7)
V_R_SLICE = slice(7, 10)
IMAGE_SLICE = slice(10, 12)
GYRO_BIAS_SLICE = slice(12, 15)
ACCEL_BIAS_SLICE = slice(15, 18)


@dataclass(frozen=True)
class InterceptionState18:
    """Named representation of x=[q,p_r,v_r,p_bar_i,b_gyr,b_acc]."""

    q: Tuple[float, float, float, float]
    p_r_e: Tuple[float, float, float]
    v_r_e: Tuple[float, float, float]
    image_xy: Tuple[float, float]
    b_gyr_b: Tuple[float, float, float]
    b_acc_b: Tuple[float, float, float]

    def as_vector(self) -> np.ndarray:
        """Return the paper-ordered 18-element state vector."""
        vector = np.asarray(
            self.q
            + self.p_r_e
            + self.v_r_e
            + self.image_xy
            + self.b_gyr_b
            + self.b_acc_b,
            dtype=float,
        )
        if vector.shape != (STATE_DIM,) or not np.all(np.isfinite(vector)):
            raise ValueError('observer state must contain 18 finite values')
        return vector

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> 'InterceptionState18':
        """Build a named state from the paper-ordered vector."""
        values = np.asarray(vector, dtype=float)
        if values.shape != (STATE_DIM,) or not np.all(np.isfinite(values)):
            raise ValueError('observer state vector must be finite and length 18')
        return cls(
            q=tuple(float(value) for value in values[Q_SLICE]),
            p_r_e=tuple(float(value) for value in values[P_R_SLICE]),
            v_r_e=tuple(float(value) for value in values[V_R_SLICE]),
            image_xy=tuple(float(value) for value in values[IMAGE_SLICE]),
            b_gyr_b=tuple(float(value) for value in values[GYRO_BIAS_SLICE]),
            b_acc_b=tuple(float(value) for value in values[ACCEL_BIAS_SLICE]),
        )


@dataclass(frozen=True)
class ObserverNoise:
    """Initial uncertainty and white-noise standard deviations."""

    initial_q_std: float = 0.005
    initial_position_std_m: float = 0.5
    initial_velocity_std_m_s: float = 0.2
    initial_image_std: float = 0.02
    initial_gyro_bias_std_rad_s: float = 0.005
    initial_accel_bias_std_m_s2: float = 0.05
    gyro_noise_std_rad_s: float = 0.015
    accel_noise_std_m_s2: float = 0.15
    image_noise_std: float = 0.015

    def validate(self) -> None:
        """Reject nonfinite or nonpositive uncertainty definitions."""
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class ObserverConfig:
    """Frame geometry and numerical limits for the paper observer."""

    gravity_e: Tuple[float, float, float] = (0.0, 0.0, -9.80665)
    camera_to_body_rotation: Tuple[float, ...] = (
        0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
    )
    minimum_depth_m: float = 0.25
    maximum_dt_s: float = 0.05
    dkf_delay_steps: int = 0

    def validate(self) -> None:
        """Validate frames and finite numerical limits."""
        gravity = np.asarray(self.gravity_e, dtype=float)
        rotation = np.asarray(self.camera_to_body_rotation, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError('gravity_e must be a finite three-vector')
        if rotation.shape != (9,) or not np.all(np.isfinite(rotation)):
            raise ValueError('camera_to_body_rotation must contain 9 values')
        rotation = rotation.reshape((3, 3))
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError('camera_to_body_rotation must be orthonormal')
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
            raise ValueError('camera_to_body_rotation must have determinant +1')
        for name in (
            'minimum_depth_m',
            'maximum_dt_s',
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        value = self.dkf_delay_steps
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('dkf_delay_steps must be a nonnegative integer')


@dataclass(frozen=True)
class ObserverSnapshot:
    """Immutable observer output and diagnostics."""

    state: InterceptionState18
    covariance: np.ndarray
    prediction_count: int
    correction_count: int
    innovation_norm: float


@dataclass
class _PredictionRecord:
    """One saved IMU transition used to replay a delayed correction."""

    prior_x: np.ndarray
    prior_covariance: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    dt_s: float


@dataclass
class _DkfCycleRecord:
    """State checkpoint and IMU substeps belonging to one DKF period."""

    prior_x: np.ndarray
    prior_covariance: np.ndarray
    predictions: list[_PredictionRecord]


class PaperStateObserver:
    """DKF using the paper's Appendix B propagation and delayed update."""

    def __init__(
        self,
        config: ObserverConfig = ObserverConfig(),
        noise: ObserverNoise = ObserverNoise(),
    ) -> None:
        config.validate()
        noise.validate()
        self.config = config
        self.noise = noise
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.prediction_count = 0
        self.correction_count = 0
        self.innovation_norm = 0.0
        self.last_F = np.eye(STATE_DIM)
        self.last_G = np.zeros((STATE_DIM, PROCESS_NOISE_DIM))
        self._dkf_history: list[_DkfCycleRecord] = []
        self._pending_predictions: list[_PredictionRecord] = []
        self._pending_prior_x: np.ndarray | None = None
        self._pending_prior_covariance: np.ndarray | None = None
        self._gravity = np.asarray(config.gravity_e, dtype=float)
        self._r_c_to_b = np.asarray(
            config.camera_to_body_rotation,
            dtype=float,
        ).reshape((3, 3))
        self._Q = np.diag(
            [noise.gyro_noise_std_rad_s ** 2] * 3
            + [noise.accel_noise_std_m_s2 ** 2] * 3
        )
        self._R = np.eye(MEASUREMENT_DIM) * noise.image_noise_std ** 2

    @property
    def initialized(self) -> bool:
        """Return whether an initial state and covariance are available."""
        return self.x is not None and self.P is not None

    def reset(self) -> None:
        """Discard the estimate so the next image establishes a new x(0)."""
        self.x = None
        self.P = None
        self.prediction_count = 0
        self.correction_count = 0
        self.innovation_norm = 0.0
        self.last_F = np.eye(STATE_DIM)
        self.last_G = np.zeros((STATE_DIM, PROCESS_NOISE_DIM))
        self._clear_delay_history()

    def initialize(self, state: InterceptionState18) -> ObserverSnapshot:
        """Initialize x and diagonal P without any target world-state input."""
        vector = state.as_vector()
        vector[Q_SLICE] = _normalize_quaternion(vector[Q_SLICE])
        std = self.noise
        standard_deviations = np.array(
            [std.initial_q_std] * 4
            + [std.initial_position_std_m] * 3
            + [std.initial_velocity_std_m_s] * 3
            + [std.initial_image_std] * 2
            + [std.initial_gyro_bias_std_rad_s] * 3
            + [std.initial_accel_bias_std_m_s2] * 3,
            dtype=float,
        )
        self.x = vector
        self.P = np.diag(standard_deviations ** 2)
        self.prediction_count = 0
        self.correction_count = 0
        self.innovation_norm = 0.0
        self.last_F = np.eye(STATE_DIM)
        self.last_G = np.zeros((STATE_DIM, PROCESS_NOISE_DIM))
        self._clear_delay_history()
        return self.snapshot()

    def initialize_from_image(
        self,
        q_b_to_e: Sequence[float],
        image_xy: Sequence[float],
        camera_depth_m: float,
        initial_v_r_e: Sequence[float] = (0.0, 0.0, 0.0),
        initial_b_gyr_b: Sequence[float] = (0.0, 0.0, 0.0),
        initial_b_acc_b: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> ObserverSnapshot:
        """Initialize p_r from image LOS and an engineering depth prior."""
        q = _normalize_quaternion(_finite_vector(q_b_to_e, 4, 'q_b_to_e'))
        image = _finite_vector(image_xy, 2, 'image_xy')
        if not math.isfinite(camera_depth_m) or camera_depth_m <= 0.0:
            raise ValueError('camera_depth_m must be finite and positive')
        target_c = np.array(
            (image[0] * camera_depth_m, image[1] * camera_depth_m,
             camera_depth_m),
            dtype=float,
        )
        r_b_to_e = _quaternion_to_rotation(q)
        # p_r = p_interceptor - p_target, hence the leading minus sign.
        p_r_e = -(r_b_to_e @ self._r_c_to_b @ target_c)
        state = InterceptionState18(
            q=tuple(q),
            p_r_e=tuple(float(value) for value in p_r_e),
            v_r_e=tuple(_finite_vector(initial_v_r_e, 3, 'initial_v_r_e')),
            image_xy=tuple(image),
            b_gyr_b=tuple(
                _finite_vector(initial_b_gyr_b, 3, 'initial_b_gyr_b')
            ),
            b_acc_b=tuple(
                _finite_vector(initial_b_acc_b, 3, 'initial_b_acc_b')
            ),
        )
        return self.initialize(state)

    def predict(
        self,
        gyro_rad_s_b: Sequence[float],
        accel_m_s2_b: Sequence[float],
        dt_s: float,
    ) -> ObserverSnapshot:
        """Apply paper (44)-(55), then covariance equations (30)-(31)."""
        x, covariance = self._require_state()
        gyro = _finite_vector(gyro_rad_s_b, 3, 'gyro_rad_s_b')
        accel = _finite_vector(accel_m_s2_b, 3, 'accel_m_s2_b')
        if (
            not math.isfinite(dt_s)
            or dt_s <= 0.0
            or dt_s > self.config.maximum_dt_s
        ):
            raise ValueError(
                f'dt_s must be within (0, {self.config.maximum_dt_s}]'
            )

        if self.config.dkf_delay_steps > 0:
            if self._pending_prior_x is None:
                self._pending_prior_x = x.copy()
                self._pending_prior_covariance = covariance.copy()
            self._pending_predictions.append(
                _PredictionRecord(
                    prior_x=x.copy(),
                    prior_covariance=covariance.copy(),
                    gyro=gyro.copy(),
                    accel=accel.copy(),
                    dt_s=dt_s,
                )
            )
        predicted, predicted_covariance, F, G = self._predict_arrays(
            x,
            covariance,
            gyro,
            accel,
            dt_s,
        )
        self.x = predicted
        self.P = predicted_covariance
        self.last_F = F
        self.last_G = G
        self.prediction_count += 1
        return self.snapshot()

    def correct_image(self, image_xy: Sequence[float]) -> ObserverSnapshot:
        """Apply the current-image D=0 form of paper equations (32)-(36)."""
        return self._correct_current_image(image_xy)

    @property
    def delay_history_ready(self) -> bool:
        """Return whether all D saved DKF periods are available."""
        delay_steps = self.config.dkf_delay_steps
        return delay_steps == 0 or len(self._dkf_history) >= delay_steps

    def complete_dkf_cycle(self) -> None:
        """Save one 50 Hz DKF checkpoint containing its IMU substeps."""
        delay_steps = self.config.dkf_delay_steps
        if delay_steps == 0:
            return
        x, covariance = self._require_state()
        prior_x = self._pending_prior_x
        prior_covariance = self._pending_prior_covariance
        if prior_x is None or prior_covariance is None:
            prior_x = x.copy()
            prior_covariance = covariance.copy()
        self._dkf_history.append(
            _DkfCycleRecord(
                prior_x=prior_x,
                prior_covariance=prior_covariance,
                predictions=self._pending_predictions,
            )
        )
        excess = len(self._dkf_history) - delay_steps
        if excess > 0:
            del self._dkf_history[:excess]
        self._pending_predictions = []
        self._pending_prior_x = None
        self._pending_prior_covariance = None

    def correct_delayed_image(
        self,
        image_xy: Sequence[float],
    ) -> ObserverSnapshot:
        """Correct x[k-D] and replay D stored DKF blocks per Algorithm 2."""
        delay_steps = self.config.dkf_delay_steps
        if delay_steps == 0:
            return self._correct_current_image(image_xy)
        if self._pending_predictions:
            raise RuntimeError(
                'complete_dkf_cycle must be called before delayed correction'
            )
        if len(self._dkf_history) < delay_steps:
            raise RuntimeError(
                'delayed image history is not ready: '
                f'{len(self._dkf_history)}/{delay_steps} DKF cycles'
            )

        current_x, current_covariance = self._require_state()
        saved_x = current_x.copy()
        saved_covariance = current_covariance.copy()
        saved_history = [
            _DkfCycleRecord(
                prior_x=cycle.prior_x.copy(),
                prior_covariance=cycle.prior_covariance.copy(),
                predictions=[
                    _PredictionRecord(
                        prior_x=record.prior_x.copy(),
                        prior_covariance=record.prior_covariance.copy(),
                        gyro=record.gyro.copy(),
                        accel=record.accel.copy(),
                        dt_s=record.dt_s,
                    )
                    for record in cycle.predictions
                ],
            )
            for cycle in self._dkf_history
        ]
        saved_correction_count = self.correction_count
        saved_innovation_norm = self.innovation_norm
        saved_last_F = self.last_F.copy()
        saved_last_G = self.last_G.copy()

        try:
            oldest = self._dkf_history[0]
            self.x = oldest.prior_x.copy()
            self.P = oldest.prior_covariance.copy()
            self._correct_current_image(image_xy)

            replay_x, replay_covariance = self._require_state()
            for cycle in self._dkf_history:
                cycle.prior_x = replay_x.copy()
                cycle.prior_covariance = replay_covariance.copy()
                for record in cycle.predictions:
                    record.prior_x = replay_x.copy()
                    record.prior_covariance = replay_covariance.copy()
                    (
                        replay_x,
                        replay_covariance,
                        self.last_F,
                        self.last_G,
                    ) = self._predict_arrays(
                        replay_x,
                        replay_covariance,
                        record.gyro,
                        record.accel,
                        record.dt_s,
                    )
            self.x = replay_x
            self.P = replay_covariance
            return self.snapshot()
        except Exception:
            # A failed replay must be atomic: retain the live current estimate
            # and the exact checkpoints needed by the next delayed frame.
            self.x = saved_x
            self.P = saved_covariance
            self._dkf_history = saved_history
            self.correction_count = saved_correction_count
            self.innovation_norm = saved_innovation_norm
            self.last_F = saved_last_F
            self.last_G = saved_last_G
            raise

    def _clear_delay_history(self) -> None:
        """Clear completed DKF blocks and an in-progress block."""
        self._dkf_history.clear()
        self._pending_predictions.clear()
        self._pending_prior_x = None
        self._pending_prior_covariance = None

    def _correct_current_image(
        self,
        image_xy: Sequence[float],
    ) -> ObserverSnapshot:
        """Apply equations (32)-(36) to the estimate at its current epoch."""
        x, covariance = self._require_state()
        measurement = _finite_vector(image_xy, 2, 'image_xy')
        H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        H[:, IMAGE_SLICE] = np.eye(MEASUREMENT_DIM)
        innovation = measurement - H @ x
        innovation_covariance = H @ covariance @ H.T + self._R
        self.innovation_norm = float(np.linalg.norm(innovation))
        # Solve the right-sided system without forming an explicit inverse:
        # K = P H^T S^{-1}, exactly as in paper equation (36).
        gain = np.linalg.solve(
            innovation_covariance.T,
            (covariance @ H.T).T,
        ).T
        corrected = x + gain @ innovation
        corrected[Q_SLICE] = _normalize_quaternion(
            corrected[Q_SLICE],
            reference=x[Q_SLICE],
        )
        # Keep the covariance update exactly as written in paper equation
        # (35).  The Joseph form and covariance regularization are not part of
        # the paper's DKF algorithm.
        corrected_covariance = (np.eye(STATE_DIM) - gain @ H) @ covariance
        self.x = corrected
        self.P = corrected_covariance
        self.correction_count += 1
        return self.snapshot()

    def _predict_arrays(
        self,
        x: np.ndarray,
        covariance: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict arrays without changing counters or recording history."""
        predicted, F, G = self._appendix_prediction(x, gyro, accel, dt_s)
        predicted_covariance = F @ covariance @ F.T + G @ self._Q @ G.T
        return predicted, predicted_covariance, F, G

    def snapshot(self) -> ObserverSnapshot:
        """Return defensive copies of the current state and covariance."""
        x, covariance = self._require_state()
        return ObserverSnapshot(
            state=InterceptionState18.from_vector(x),
            covariance=covariance.copy(),
            prediction_count=self.prediction_count,
            correction_count=self.correction_count,
            innovation_norm=self.innovation_norm,
        )

    def _require_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.x is None or self.P is None:
            raise RuntimeError('observer is not initialized')
        return self.x, self.P

    def _appendix_prediction(
        self,
        x: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the nonlinear recurrence and Appendix Jacobian blocks."""
        q = x[Q_SLICE]
        p_r = x[P_R_SLICE]
        v_r = x[V_R_SLICE]
        image = x[IMAGE_SLICE]
        b_gyr = x[GYRO_BIAS_SLICE]
        b_acc = x[ACCEL_BIAS_SLICE]
        omega_b = gyro - b_gyr
        specific_force_b = accel - b_acc
        r_b_to_e = _quaternion_to_rotation(q)
        target_c = self._r_c_to_b.T @ r_b_to_e.T @ (-p_r)
        depth_m = float(target_c[2])
        if (
            not math.isfinite(depth_m)
            or depth_m < self.config.minimum_depth_m
        ):
            raise ValueError('estimated target depth is invalid or behind camera')

        delta_matrix = _appendix_delta_quaternion_matrix(omega_b, dt_s)
        q_new = _normalize_quaternion(delta_matrix @ q, reference=q)
        acceleration_e = r_b_to_e @ specific_force_b + self._gravity
        v_r_new = v_r + acceleration_e * dt_s
        p_r_new = p_r + 0.5 * (v_r + v_r_new) * dt_s

        velocity_c = self._r_c_to_b.T @ r_b_to_e.T @ v_r
        omega_c = self._r_c_to_b.T @ omega_b
        translation_jacobian = _translation_image_jacobian(image, depth_m)
        rotation_jacobian = _rotation_image_jacobian(image)
        image_new = image + (
            translation_jacobian @ velocity_c
            + rotation_jacobian @ omega_c
        ) * dt_s

        predicted = x.copy()
        predicted[Q_SLICE] = q_new
        predicted[P_R_SLICE] = p_r_new
        predicted[V_R_SLICE] = v_r_new
        predicted[IMAGE_SLICE] = image_new

        F = np.eye(STATE_DIM)
        G = np.zeros((STATE_DIM, PROCESS_NOISE_DIM))
        F[Q_SLICE, Q_SLICE] = delta_matrix
        f_q_b_gyr = _appendix_quaternion_bias_jacobian(q, dt_s)
        F[Q_SLICE, GYRO_BIAS_SLICE] = f_q_b_gyr
        G[Q_SLICE, 0:3] = f_q_b_gyr
        F[P_R_SLICE, V_R_SLICE] = np.eye(3) * dt_s

        rotation_derivatives = _rotation_derivatives(q)
        f_v_q = np.column_stack(
            [derivative @ specific_force_b for derivative in rotation_derivatives]
        ) * dt_s
        F[V_R_SLICE, Q_SLICE] = f_v_q
        f_v_b_acc = -r_b_to_e * dt_s
        F[V_R_SLICE, ACCEL_BIAS_SLICE] = f_v_b_acc
        G[V_R_SLICE, 3:6] = f_v_b_acc

        f_image_q = np.column_stack(
            [
                translation_jacobian
                @ self._r_c_to_b.T
                @ derivative.T
                @ v_r
                for derivative in rotation_derivatives
            ]
        ) * dt_s
        F[IMAGE_SLICE, Q_SLICE] = f_image_q
        F[IMAGE_SLICE, V_R_SLICE] = (
            translation_jacobian @ self._r_c_to_b.T @ r_b_to_e.T * dt_s
        )
        F[IMAGE_SLICE, IMAGE_SLICE] = _appendix_image_state_jacobian(
            image,
            velocity_c,
            omega_c,
            depth_m,
            dt_s,
        )
        # Equation (55) follows from the discretized equation (51); dt is
        # required dimensionally even though it is omitted in the typeset (55).
        f_image_b_gyr = (
            -rotation_jacobian @ self._r_c_to_b.T * dt_s
        )
        F[IMAGE_SLICE, GYRO_BIAS_SLICE] = f_image_b_gyr
        G[IMAGE_SLICE, 0:3] = f_image_b_gyr
        return predicted, F, G


def _appendix_delta_quaternion_matrix(
    omega_b: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return Appendix C M(delta-q) for equation (44)."""
    x, y, z = 0.5 * omega_b * dt_s
    return np.array(
        (
            (1.0, -x, -y, -z),
            (x, 1.0, z, -y),
            (y, -z, 1.0, x),
            (z, y, -x, 1.0),
        ),
        dtype=float,
    )


def _appendix_quaternion_bias_jacobian(
    q: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return equation (45), F(q_k,b_gyr[k-1])."""
    q0, q1, q2, q3 = q
    return 0.5 * dt_s * np.array(
        (
            (q1, q2, q3),
            (-q0, q3, -q2),
            (-q3, -q0, q1),
            (q2, -q1, -q0),
        ),
        dtype=float,
    )


def _translation_image_jacobian(
    image_xy: np.ndarray,
    depth_m: float,
) -> np.ndarray:
    """Return the translational block in equations (51) and (53)."""
    x, y = image_xy
    inverse_depth = 1.0 / depth_m
    return np.array(
        (
            (-inverse_depth, 0.0, x * inverse_depth),
            (0.0, -inverse_depth, y * inverse_depth),
        ),
        dtype=float,
    )


def _rotation_image_jacobian(image_xy: np.ndarray) -> np.ndarray:
    """Return the rotational IBVS block in equations (51) and (55)."""
    x, y = image_xy
    return np.array(
        (
            (x * y, -(1.0 + x * x), y),
            (1.0 + y * y, -x * y, -x),
        ),
        dtype=float,
    )


def _appendix_image_state_jacobian(
    image_xy: np.ndarray,
    velocity_c: np.ndarray,
    omega_c: np.ndarray,
    depth_m: float,
    dt_s: float,
) -> np.ndarray:
    """Return equation (54), F(image_k,image[k-1])."""
    x, y = image_xy
    _, _, velocity_z = velocity_c
    omega_x, omega_y, omega_z = omega_c
    return np.eye(2) + np.array(
        (
            (
                velocity_z / depth_m + y * omega_x - 2.0 * x * omega_y,
                x * omega_x + omega_z,
            ),
            (
                -y * omega_y - omega_z,
                velocity_z / depth_m + 2.0 * y * omega_x - x * omega_y,
            ),
        ),
        dtype=float,
    ) * dt_s


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Return R_b^e in the full quadratic form used by Appendix C."""
    q0, q1, q2, q3 = q
    return np.array(
        (
            (
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ),
            (
                2.0 * (q1 * q2 + q0 * q3),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2.0 * (q2 * q3 - q0 * q1),
            ),
            (
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ),
        ),
        dtype=float,
    )


def _rotation_derivatives(q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return d(R_b^e)/dq, equivalent to Appendix C M1-M5 blocks."""
    q0, q1, q2, q3 = q
    return (
        2.0 * np.array(
            ((q0, -q3, q2), (q3, q0, -q1), (-q2, q1, q0))
        ),
        2.0 * np.array(
            ((q1, q2, q3), (q2, -q1, -q0), (q3, q0, -q1))
        ),
        2.0 * np.array(
            ((-q2, q1, q0), (q1, q2, q3), (-q0, q3, -q2))
        ),
        2.0 * np.array(
            ((-q3, -q0, q1), (q0, -q3, q2), (q1, q2, q3))
        ),
    )


def _normalize_quaternion(
    quaternion: Sequence[float],
    reference: Sequence[float] | None = None,
) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError('quaternion must contain four finite values')
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError('quaternion norm must be nonzero')
    normalized = values / norm
    if reference is not None and float(np.dot(normalized, reference)) < 0.0:
        normalized = -normalized
    return normalized


def _finite_vector(
    values: Sequence[float],
    length: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain {length} finite values')
    return result
