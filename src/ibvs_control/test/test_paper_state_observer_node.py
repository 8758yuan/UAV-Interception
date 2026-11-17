"""ROS-message integration test for the paper state observer node."""

from copy import deepcopy

import numpy as np
from px4_msgs.msg import SensorCombined, TimesyncStatus, VehicleAttitude
import rclpy
from std_msgs.msg import Empty

from interception_interfaces.msg import VisionFeature
from ibvs_control.paper_state_observer_node import PaperStateObserverNode


def test_node_waits_for_interception_reset_then_updates_online(
    monkeypatch,
) -> None:
    """Skip invalid images while predicting, then resume delayed updates."""
    rclpy.init()
    node = None
    try:
        node = PaperStateObserverNode()
        published = []
        monkeypatch.setattr(node.publisher, 'publish', published.append)
        assert node.observer.config.camera_position_b_m == (0.14, 0.0, 0.0)
        assert node.observer.noise.target_accel_noise_density_m2_s3 == (
            node.get_parameter('target_accel_noise_density_m2_s3').value
        )
        node.require_range_bootstrap = False
        attitude = VehicleAttitude()
        attitude.q = [1.0, 0.0, 0.0, 0.0]
        node._attitude_callback(attitude)

        feature = VisionFeature()
        feature.capture_stamp.sec = 1
        feature.valid = True
        feature.x_norm = 0.02
        feature.y_norm = -0.01
        node._feature_callback(feature)
        assert node.image_update_count == 1
        assert not node.observer.initialized
        assert not node.initialization_armed

        node._reset_callback(Empty())
        node._feature_callback(feature)
        assert node.image_update_count == 2
        assert node.observer.initialized
        assert node.observer.prediction_count == 0

        epoch_us = 1_790_000_000_000_000
        sync = TimesyncStatus()
        sync.source_protocol = TimesyncStatus.SOURCE_PROTOCOL_DDS
        sync.estimated_offset = -epoch_us
        node._timesync_callback(sync)
        timestamp_us = epoch_us + 1_000_000
        for cycle in range(4):
            sample_count = 5 if cycle == 0 else 4
            for _ in range(sample_count):
                imu = SensorCombined()
                imu.timestamp = timestamp_us
                imu.gyro_rad = [0.0, 0.0, 0.0]
                # Stationary PX4 FRD specific force becomes +g in ROS FLU.
                imu.accelerometer_m_s2 = [0.0, 0.0, -9.80665]
                node._imu_callback(imu)
                timestamp_us += 5_000
            if cycle == 3:
                feature.x_norm = 0.021
                feature.capture_stamp.nanosec = 15_000_000
                node._feature_callback(feature)
            node._dkf_callback()

        assert node.image_update_count == 3
        snapshot = node.observer.snapshot()
        assert snapshot.prediction_count == 16
        assert snapshot.correction_count == 1
        assert np.linalg.norm(snapshot.state.q) == 1.0
        assert np.all(np.isfinite(snapshot.state.as_vector()))
        assert np.all(np.isfinite(snapshot.covariance))
        assert len(published[-1].covariance) == 324
        state = snapshot.state.as_vector()
        projected, H = node.observer._project_image(state)
        np.testing.assert_allclose(published[-1].image_xy, projected)
        np.testing.assert_allclose(
            snapshot.covariance[10:12], H @ snapshot.covariance, atol=1e-12,
        )

        reference = deepcopy(node.observer)
        invalid = VisionFeature()
        invalid.valid = False
        invalid.reason = 'target_class_not_detected'
        invalid.capture_stamp.sec = 1
        invalid.capture_stamp.nanosec = 50_000_000
        invalid.x_norm = float('nan')
        invalid.y_norm = float('nan')
        feature.capture_stamp.nanosec = 75_000_000

        for frame, correction_count in ((invalid, 1), (feature, 2)):
            for _ in range(4):
                imu = SensorCombined()
                imu.timestamp = timestamp_us
                imu.gyro_rad = [0.0, 0.0, 0.1]
                imu.accelerometer_m_s2 = [0.0, 0.0, -9.80665]
                node._imu_callback(imu)
                timestamp_us += 5_000
                reference.predict(
                    np.asarray(imu.gyro_rad, dtype=float) * (1, -1, -1),
                    np.asarray(imu.accelerometer_m_s2, dtype=float)
                    * (1, -1, -1),
                    0.005,
                    timestamp_ns=(int(imu.timestamp) - 5000) * 1000,
                )
            reference.complete_dkf_cycle()
            if frame.valid:
                reference.correct_delayed_image(
                    (frame.x_norm, frame.y_norm),
                    capture_timestamp_ns=(
                        (epoch_us + 1_000_000) * 1000
                        + frame.capture_stamp.nanosec
                    ),
                )
            else:
                reference.discard_history_before(
                    (epoch_us + 1_000_000) * 1000
                    + frame.capture_stamp.nanosec
                )
            node._feature_callback(frame)
            node._dkf_callback()

            snapshot = node.observer.snapshot()
            assert published[-1].valid
            assert published[-1].initialized
            assert snapshot.correction_count == correction_count
            assert snapshot.prediction_count == reference.prediction_count
            np.testing.assert_allclose(
                snapshot.state.as_vector(),
                reference.snapshot().state.as_vector(),
            )
            np.testing.assert_allclose(snapshot.covariance, reference.P)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_capture_update_waits_for_dds_mapping_and_matching_imu(monkeypatch):
    rclpy.init()
    node = PaperStateObserverNode()
    try:
        messages = []
        monkeypatch.setattr(node.publisher, 'publish', messages.append)
        node.require_range_bootstrap = False
        node._attitude_callback(VehicleAttitude(q=[1.0, 0.0, 0.0, 0.0]))
        node._reset_callback(Empty())
        frame = VisionFeature()
        frame.valid = True
        frame.capture_stamp.sec = 1
        node._feature_callback(frame)
        for timestamp in (1_000_000, 1_005_000, 1_010_000):
            imu = SensorCombined()
            imu.timestamp = timestamp
            imu.accelerometer_m_s2 = [0.0, 0.0, -9.80665]
            node._imu_callback(imu)
        frame.capture_stamp.nanosec = 15_000_000
        node._feature_callback(frame)
        node._dkf_callback()
        assert messages[-1].reason == 'waiting_for_imu_clock_mapping'
        assert messages[-1].valid
        assert node.pending_feature is frame

        sync = TimesyncStatus()
        sync.source_protocol = TimesyncStatus.SOURCE_PROTOCOL_MAVLINK
        node._timesync_callback(sync)
        assert node.timesync_offset_us is None
        sync.source_protocol = TimesyncStatus.SOURCE_PROTOCOL_DDS
        node._timesync_callback(sync)
        node._dkf_callback()
        assert messages[-1].reason == 'waiting_for_capture_imu'
        assert messages[-1].valid
        assert node.observer.correction_count == 0
        imu.timestamp = 1_015_000
        node._imu_callback(imu)
        node._dkf_callback()
        assert messages[-1].reason == 'image_corrected_capture_time'
        assert node.observer.correction_count == 1
        assert node.observer.prediction_count == 3
    finally:
        node.destroy_node()
        rclpy.shutdown()
