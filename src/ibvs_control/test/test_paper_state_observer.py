"""Verification of the paper-aligned delayed 18-state observer."""

import math

import numpy as np
import pytest

from ibvs_control.paper_state_observer import (
    ACCEL_BIAS_SLICE,
    GYRO_BIAS_SLICE,
    IMAGE_SLICE,
    P_R_SLICE,
    Q_SLICE,
    STATE_DIM,
    V_R_SLICE,
    InterceptionState18,
    ObserverConfig,
    PaperStateObserver,
    _appendix_delta_quaternion_matrix,
    _appendix_image_state_jacobian,
    _quaternion_to_rotation,
    _rotation_derivatives,
)


GRAVITY = 9.80665


def _initialized_observer(
    image_xy=(0.0, 0.0),
    depth_m=12.0,
) -> PaperStateObserver:
    observer = PaperStateObserver()
    observer.initialize_from_image(
        (1.0, 0.0, 0.0, 0.0),
        image_xy,
        depth_m,
    )
    return observer


def test_state_layout_is_exactly_the_paper_18_vector() -> None:
    state = InterceptionState18(
        q=(1.0, 0.0, 0.0, 0.0),
        p_r_e=(-12.0, 1.0, 2.0),
        v_r_e=(0.1, 0.2, 0.3),
        image_xy=(0.01, -0.02),
        b_gyr_b=(0.001, 0.002, 0.003),
        b_acc_b=(0.01, 0.02, 0.03),
    )
    vector = state.as_vector()
    assert vector.shape == (STATE_DIM,)
    assert vector[Q_SLICE] == pytest.approx(state.q)
    assert vector[P_R_SLICE] == pytest.approx(state.p_r_e)
    assert vector[V_R_SLICE] == pytest.approx(state.v_r_e)
    assert vector[IMAGE_SLICE] == pytest.approx(state.image_xy)
    assert vector[GYRO_BIAS_SLICE] == pytest.approx(state.b_gyr_b)
    assert vector[ACCEL_BIAS_SLICE] == pytest.approx(state.b_acc_b)
    assert InterceptionState18.from_vector(vector) == state


def test_image_and_depth_prior_initialize_relative_position() -> None:
    observer = _initialized_observer(image_xy=(0.1, -0.05), depth_m=10.0)
    state = observer.snapshot().state
    # Optical [right, down, forward] maps to FLU [-left, -up, forward].
    assert state.p_r_e == pytest.approx((-10.0, 1.0, -0.5))
    assert state.image_xy == pytest.approx((0.1, -0.05))
    assert np.linalg.norm(state.q) == pytest.approx(1.0)


def test_stationary_imu_prediction_preserves_mean_and_is_finite() -> None:
    observer = _initialized_observer()
    initial = observer.snapshot()
    result = observer.predict((0.0, 0.0, 0.0), (0.0, 0.0, GRAVITY), 0.01)
    assert result.state.q == pytest.approx(initial.state.q)
    assert result.state.p_r_e == pytest.approx(initial.state.p_r_e)
    assert result.state.v_r_e == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert result.state.image_xy == pytest.approx((0.0, 0.0))
    assert result.prediction_count == 1
    assert np.all(np.isfinite(result.covariance))
    assert np.linalg.eigvalsh(result.covariance)[0] >= -1e-12


def test_quaternion_recurrence_matches_appendix_c_matrix() -> None:
    observer = _initialized_observer()
    omega = np.array((0.1, -0.2, 0.3))
    dt_s = 0.01
    expected = _appendix_delta_quaternion_matrix(omega, dt_s) @ np.array(
        (1.0, 0.0, 0.0, 0.0)
    )
    expected /= np.linalg.norm(expected)
    result = observer.predict(omega, (0.0, 0.0, GRAVITY), dt_s)
    assert result.state.q == pytest.approx(expected)
    assert np.linalg.norm(result.state.q) == pytest.approx(1.0)


def test_velocity_quaternion_block_matches_equation_49() -> None:
    q = np.array((0.8, 0.2, -0.3, 0.4), dtype=float)
    q /= np.linalg.norm(q)
    acceleration_b = np.array((0.3, -0.8, 1.2))
    derivatives = _rotation_derivatives(q)
    analytic = np.column_stack(
        [derivative @ acceleration_b for derivative in derivatives]
    )
    numerical = np.zeros((3, 4))
    epsilon = 1e-7
    for index in range(4):
        delta = np.zeros(4)
        delta[index] = epsilon
        numerical[:, index] = (
            _quaternion_to_rotation(q + delta) @ acceleration_b
            - _quaternion_to_rotation(q - delta) @ acceleration_b
        ) / (2.0 * epsilon)
    assert analytic == pytest.approx(numerical, abs=1e-8)


def test_image_state_block_matches_equation_54() -> None:
    image = np.array((0.2, -0.1))
    velocity_c = np.array((0.4, -0.2, 1.1))
    omega_c = np.array((0.03, -0.04, 0.02))
    depth_m = 8.0
    dt_s = 0.01
    jacobian = _appendix_image_state_jacobian(
        image,
        velocity_c,
        omega_c,
        depth_m,
        dt_s,
    )

    def propagate(candidate):
        x_value, y_value = candidate
        translation = np.array(
            (
                (-1.0 / depth_m, 0.0, x_value / depth_m),
                (0.0, -1.0 / depth_m, y_value / depth_m),
            )
        )
        rotation = np.array(
            (
                (x_value * y_value, -(1.0 + x_value ** 2), y_value),
                (1.0 + y_value ** 2, -x_value * y_value, -x_value),
            )
        )
        return candidate + (
            translation @ velocity_c + rotation @ omega_c
        ) * dt_s

    numerical = np.zeros((2, 2))
    epsilon = 1e-7
    for index in range(2):
        delta = np.zeros(2)
        delta[index] = epsilon
        numerical[:, index] = (
            propagate(image + delta) - propagate(image - delta)
        ) / (2.0 * epsilon)
    assert jacobian == pytest.approx(numerical, abs=1e-8)


def test_current_image_update_reduces_image_uncertainty() -> None:
    observer = _initialized_observer()
    observer.predict((0.0, 0.0, 0.0), (0.0, 0.0, GRAVITY), 0.01)
    before = observer.snapshot()
    after = observer.correct_image((0.04, -0.03))
    assert after.state.image_xy[0] > before.state.image_xy[0]
    assert after.state.image_xy[1] < before.state.image_xy[1]
    assert after.correction_count == 1
    assert after.innovation_norm == pytest.approx(0.05)
    assert np.trace(after.covariance[IMAGE_SLICE, IMAGE_SLICE]) < np.trace(
        before.covariance[IMAGE_SLICE, IMAGE_SLICE]
    )


def test_delayed_update_corrects_past_state_then_replays_imu_substeps() -> None:
    """D counts DKF periods, independent of IMU substeps in each period."""
    config = ObserverConfig(dkf_delay_steps=2)
    delayed = PaperStateObserver(config=config)
    reference = PaperStateObserver(config=config)
    for observer in (delayed, reference):
        observer.initialize_from_image(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0),
            12.0,
        )

    imu_samples = [
        ((0.01, -0.02, 0.03), (0.1, -0.1, GRAVITY), 0.005),
        ((0.02, -0.01, 0.02), (0.0, 0.1, GRAVITY), 0.005),
        ((0.03, 0.00, 0.01), (-0.1, 0.0, GRAVITY), 0.005),
        ((0.02, 0.01, 0.00), (0.0, -0.1, GRAVITY), 0.005),
    ]
    measurement = (0.01, -0.008)
    reference.correct_image(measurement)
    for index, (gyro, accel, dt_s) in enumerate(imu_samples):
        reference.predict(gyro, accel, dt_s)
        delayed.predict(gyro, accel, dt_s)
        # Deliberately make the two DKF periods contain one and three IMU
        # substeps: readiness must depend on two periods, not four samples.
        if index in (0, 3):
            delayed.complete_dkf_cycle()

    assert delayed.delay_history_ready
    result = delayed.correct_delayed_image(measurement)
    expected = reference.snapshot()
    assert result.state.as_vector() == pytest.approx(
        expected.state.as_vector(), abs=1e-12
    )
    assert result.covariance == pytest.approx(
        expected.covariance, abs=1e-12
    )
    assert result.prediction_count == len(imu_samples)
    assert result.correction_count == 1


def test_delayed_update_waits_for_complete_dkf_history() -> None:
    observer = PaperStateObserver(
        config=ObserverConfig(dkf_delay_steps=4)
    )
    observer.initialize_from_image(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0),
        12.0,
    )
    for _ in range(3):
        for _ in range(4):
            observer.predict(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, GRAVITY),
                0.005,
            )
        observer.complete_dkf_cycle()
    before = observer.snapshot()
    assert not observer.delay_history_ready
    with pytest.raises(RuntimeError, match='3/4 DKF cycles'):
        observer.correct_delayed_image((0.01, 0.0))
    after = observer.snapshot()
    assert after.state == before.state
    assert after.covariance == pytest.approx(before.covariance)


def test_consecutive_delayed_updates_preserve_overlapping_history() -> None:
    """A replayed correction must update checkpoints used by the next frame."""
    config = ObserverConfig(dkf_delay_steps=2)
    delayed = PaperStateObserver(config=config)
    expected = PaperStateObserver(config=config)
    for observer in (delayed, expected):
        observer.initialize_from_image(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0),
            12.0,
        )
    imu_samples = [
        ((0.01, 0.00, 0.02), (0.0, 0.0, GRAVITY), 0.005),
        ((0.02, 0.00, 0.01), (0.1, 0.0, GRAVITY), 0.005),
        ((0.01, 0.01, 0.00), (0.0, 0.1, GRAVITY), 0.005),
        ((0.00, 0.02, 0.01), (0.0, 0.0, GRAVITY), 0.005),
        ((0.01, 0.01, 0.02), (-0.1, 0.0, GRAVITY), 0.005),
        ((0.02, 0.00, 0.01), (0.0, -0.1, GRAVITY), 0.005),
    ]
    first_measurement = (0.01, -0.008)
    second_measurement = (0.015, -0.006)

    expected.correct_image(first_measurement)
    for sample in imu_samples[:2]:
        expected.predict(*sample)
    expected.correct_image(second_measurement)
    for sample in imu_samples[2:]:
        expected.predict(*sample)

    for cycle in range(2):
        for sample in imu_samples[cycle * 2:(cycle + 1) * 2]:
            delayed.predict(*sample)
        delayed.complete_dkf_cycle()
    delayed.correct_delayed_image(first_measurement)
    for sample in imu_samples[4:]:
        delayed.predict(*sample)
    delayed.complete_dkf_cycle()
    result = delayed.correct_delayed_image(second_measurement)

    assert result.state.as_vector() == pytest.approx(
        expected.snapshot().state.as_vector(), abs=1e-12
    )
    assert result.covariance == pytest.approx(
        expected.snapshot().covariance, abs=1e-12
    )


def test_image_innovation_gate_rejects_anomalous_current_frame() -> None:
    observer = PaperStateObserver(
        config=ObserverConfig(maximum_image_innovation_nis=0.1)
    )
    observer.initialize_from_image(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0),
        12.0,
    )
    before = observer.snapshot()
    after = observer.correct_image((0.04, 0.0))
    assert after.correction_count == 0
    assert after.state.image_xy == pytest.approx(before.state.image_xy)
    assert after.innovation_norm == pytest.approx(0.04)


def test_image_innovation_updates_observable_imu_bias_states() -> None:
    """Appendix cross-covariances allow both bias estimates to be corrected."""
    observer = _initialized_observer()
    for _ in range(10):
        observer.predict(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, GRAVITY),
            0.01,
        )
    before = observer.snapshot().state
    # Stay inside the configured two-dimensional NIS gate so this test
    # exercises the accepted-update cross-covariances.
    after = observer.correct_image((0.04, -0.03)).state
    assert np.linalg.norm(after.b_gyr_b) > np.linalg.norm(before.b_gyr_b)
    assert np.linalg.norm(after.b_acc_b) > np.linalg.norm(before.b_acc_b)


def test_long_stationary_run_stays_normalized_continuous_and_finite() -> None:
    observer = _initialized_observer(image_xy=(0.02, -0.01))
    previous_q = np.asarray(observer.snapshot().state.q)
    for step in range(1000):
        snapshot = observer.predict(
            (0.001, -0.002, 0.0015),
            (0.0, 0.0, GRAVITY),
            0.005,
        )
        if step % 7 == 0:
            snapshot = observer.correct_image((0.02, -0.01))
        q = np.asarray(snapshot.state.q)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
        assert float(np.dot(q, previous_q)) > 0.999
        assert np.all(np.isfinite(snapshot.state.as_vector()))
        assert np.all(np.isfinite(snapshot.covariance))
        previous_q = q
    assert snapshot.prediction_count == 1000
    assert snapshot.correction_count == math.ceil(1000 / 7)
    assert snapshot.state.image_xy == pytest.approx((0.02, -0.01), abs=0.01)
    assert np.linalg.norm(snapshot.state.v_r_e) < 1.0
    assert np.linalg.eigvalsh(snapshot.covariance)[0] >= -1e-10


def test_invalid_prediction_does_not_mutate_state() -> None:
    observer = _initialized_observer()
    before = observer.snapshot()
    with pytest.raises(ValueError, match='dt_s'):
        observer.predict((0.0, 0.0, 0.0), (0.0, 0.0, GRAVITY), 0.1)
    after = observer.snapshot()
    assert after.state == before.state
    assert after.covariance == pytest.approx(before.covariance)


def test_reset_discards_state_and_counters() -> None:
    observer = _initialized_observer()
    observer.predict((0.0, 0.0, 0.0), (0.0, 0.0, GRAVITY), 0.01)
    observer.reset()
    assert not observer.initialized
    assert observer.prediction_count == 0
    assert observer.correction_count == 0


def test_dkf_delay_configuration_requires_integer_period_counts() -> None:
    with pytest.raises(ValueError, match='dkf_delay_steps'):
        ObserverConfig(dkf_delay_steps=-1).validate()
