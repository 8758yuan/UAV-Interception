"""Synthetic stationary example: 200 Hz IMU, 50 Hz DKF, D=4.

Run from the bundle root with PYTHONPATH=src/ibvs_control.
This checks API use; it is not a flight or closed-loop experiment.
"""

import numpy as np

from ibvs_control.paper_state_observer import (
    InterceptionState18,
    ObserverConfig,
    ObserverNoise,
    PaperStateObserver,
)


def main() -> None:
    # Match the distributed ROS configuration explicitly. The core defaults
    # alone use D=0 and a zero camera offset, and do not load YAML files.
    observer = PaperStateObserver(
        ObserverConfig(dkf_delay_steps=4, camera_position_b_m=(0.14, 0.0, 0.0)),
        ObserverNoise(
            initial_position_std_m=0.6,
            initial_velocity_std_m_s=0.6,
            accel_noise_std_m_s2=0.5,
        ),
    )
    # Known synthetic x(0): target lies 25 m along the camera optical axis;
    # p_r is measured from the body origin, including the 0.14 m camera offset.
    initial = InterceptionState18(
        q=(1.0, 0.0, 0.0, 0.0),
        p_r_e=(-25.14, 0.0, 0.0),
        v_r_e=(0.0, 0.0, 0.0),
        image_xy=(0.0, 0.0),
        b_gyr_b=(0.0, 0.0, 0.0),
        b_acc_b=(0.0, 0.0, 0.0),
    )
    observer.initialize(initial, relative_covariance=np.eye(6) * 0.6**2)

    for _ in range(4):
        # Core inputs are FLU; a stationary, level IMU measures +g on FLU z.
        for _ in range(4):
            observer.predict((0.0, 0.0, 0.0), (0.0, 0.0, 9.80665), 0.005)
        observer.complete_dkf_cycle()

    assert observer.delay_history_ready
    result = observer.correct_delayed_image((0.0, 0.0))
    np.testing.assert_allclose(result.state.as_vector(), initial.as_vector(), atol=1e-12)
    assert result.covariance.shape == (18, 18)
    assert np.all(np.isfinite(result.covariance))
    assert result.prediction_count == 16
    assert result.correction_count == 1
    print('API check passed: 16 IMU predictions, 1 delayed image correction.')
    print('state =', result.state)
    print('covariance shape =', result.covariance.shape)


if __name__ == '__main__':
    main()
