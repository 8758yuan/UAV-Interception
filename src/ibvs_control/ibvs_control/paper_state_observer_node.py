"""ROS 2 wrapper for the paper's undelayed 18-state observer."""

import math
from typing import Optional

from interception_interfaces.msg import ObserverState, VisionFeature
import numpy as np
from px4_msgs.msg import SensorCombined, VehicleAttitude
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Empty

from ibvs_control.frames import (
    frd_to_flu,
    px4_quaternion_to_enu_flu_rotation,
    rotation_to_quaternion_wxyz,
)
from ibvs_control.paper_state_observer import (
    ObserverConfig,
    ObserverNoise,
    ObserverSnapshot,
    PaperStateObserver,
)


class PaperStateObserverNode(Node):
    """Fuse onboard IMU and current image measurements without target truth."""

    def __init__(self) -> None:
        super().__init__('paper_state_observer')
        self._declare_parameters()
        self.initial_depth_m = self._positive('initial_camera_depth_m')
        self.initial_v_r_e = self._vector_parameter('initial_v_r_e', 3)
        self.initial_b_gyr_b = self._vector_parameter('initial_b_gyr_b', 3)
        self.initial_b_acc_b = self._vector_parameter('initial_b_acc_b', 3)
        config = ObserverConfig(
            gravity_e=tuple(self._vector_parameter('gravity_e', 3)),
            camera_to_body_rotation=tuple(
                self._vector_parameter('camera_to_body_rotation', 9)
            ),
            minimum_depth_m=self._positive('minimum_depth_m'),
            maximum_dt_s=self._positive('maximum_dt_s'),
            covariance_floor=self._positive('covariance_floor'),
            maximum_image_innovation_nis=self._positive(
                'maximum_image_innovation_nis'
            ),
        )
        noise = ObserverNoise(
            initial_q_std=self._positive('initial_q_std'),
            initial_position_std_m=self._positive(
                'initial_position_std_m'
            ),
            initial_velocity_std_m_s=self._positive(
                'initial_velocity_std_m_s'
            ),
            initial_image_std=self._positive('initial_image_std'),
            initial_gyro_bias_std_rad_s=self._positive(
                'initial_gyro_bias_std_rad_s'
            ),
            initial_accel_bias_std_m_s2=self._positive(
                'initial_accel_bias_std_m_s2'
            ),
            gyro_noise_std_rad_s=self._positive('gyro_noise_std_rad_s'),
            accel_noise_std_m_s2=self._positive(
                'accel_noise_std_m_s2'
            ),
            image_noise_std=self._positive('image_noise_std'),
        )
        self.observer = PaperStateObserver(config, noise)
        self.require_interception_reset = bool(
            self.get_parameter('require_interception_reset').value
        )
        # Paper x(0) belongs to the beginning of interception, after takeoff
        # and visual alignment.  Do not let pre-flight/search imagery create
        # an estimate that is already stale when rate control starts.
        self.initialization_armed = not self.require_interception_reset
        self.initial_q_b_to_e: Optional[tuple] = None
        # IMU runs faster than the camera.  Keep it as an integration buffer;
        # the public observer state is updated and published only by the image
        # callback, once per camera feature frame.
        self.pending_imu_samples: list[
            tuple[int, np.ndarray, np.ndarray]
        ] = []
        self.last_imu_timestamp_us: Optional[int] = None
        self.last_queued_imu_timestamp_us: Optional[int] = None
        self.image_update_count = 0
        self.last_error = ''

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.publisher = self.create_publisher(
            ObserverState,
            str(self.get_parameter('state_topic').value),
            10,
        )
        self.create_subscription(
            SensorCombined,
            str(self.get_parameter('imu_topic').value),
            self._imu_callback,
            sensor_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            str(self.get_parameter('attitude_topic').value),
            self._attitude_callback,
            sensor_qos,
        )
        self.create_subscription(
            VisionFeature,
            str(self.get_parameter('feature_topic').value),
            self._feature_callback,
            sensor_qos,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter('reset_topic').value),
            self._reset_callback,
            10,
        )
        self.get_logger().info(
            'Paper 18-state observer ready: image-rate update, '
            'IMU buffered between frames, D=0'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('imu_topic', '/fmu/out/sensor_combined')
        self.declare_parameter('attitude_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter(
            'feature_topic',
            '/interception/vision/raw_feature',
        )
        self.declare_parameter(
            'state_topic',
            '/interception/observer/state',
        )
        self.declare_parameter(
            'reset_topic',
            '/interception/observer/reset',
        )
        self.declare_parameter('require_interception_reset', True)
        # The 12 m depth prior matches the nominal experiment geometry. It is
        # deliberately a parameter, not a target-world-state subscription.
        self.declare_parameter('initial_camera_depth_m', 12.0)
        self.declare_parameter('initial_v_r_e', [0.0, 0.0, 0.0])
        self.declare_parameter('initial_b_gyr_b', [0.0, 0.0, 0.0])
        self.declare_parameter('initial_b_acc_b', [0.0, 0.0, 0.0])
        self.declare_parameter('gravity_e', [0.0, 0.0, -9.80665])
        self.declare_parameter(
            'camera_to_body_rotation',
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        )
        self.declare_parameter('minimum_depth_m', 0.25)
        self.declare_parameter('maximum_dt_s', 0.05)
        self.declare_parameter('covariance_floor', 1e-12)
        self.declare_parameter('maximum_image_innovation_nis', 9.21)
        self.declare_parameter('initial_q_std', 0.005)
        self.declare_parameter('initial_position_std_m', 0.5)
        self.declare_parameter('initial_velocity_std_m_s', 0.2)
        self.declare_parameter('initial_image_std', 0.02)
        self.declare_parameter('initial_gyro_bias_std_rad_s', 0.005)
        self.declare_parameter('initial_accel_bias_std_m_s2', 0.05)
        self.declare_parameter('gyro_noise_std_rad_s', 0.015)
        self.declare_parameter('accel_noise_std_m_s2', 0.15)
        self.declare_parameter('image_noise_std', 0.015)

    def _attitude_callback(self, message: VehicleAttitude) -> None:
        try:
            rotation = px4_quaternion_to_enu_flu_rotation(message.q)
            self.initial_q_b_to_e = rotation_to_quaternion_wxyz(rotation)
        except ValueError as error:
            self._log_error_once(str(error))

    def _reset_callback(self, _message: Empty) -> None:
        """Reset at the stabilized interception start, not during takeoff."""
        self.observer.reset()
        self.initialization_armed = True
        self.pending_imu_samples.clear()
        self.last_imu_timestamp_us = None
        self.last_queued_imu_timestamp_us = None
        self.last_error = ''
        self.get_logger().info(
            'Observer reset requested; waiting for the next valid image'
        )

    def _feature_callback(self, message: VisionFeature) -> None:
        self.image_update_count += 1
        image_xy = (float(message.x_norm), float(message.y_norm))
        try:
            if not self.observer.initialized:
                if not self.initialization_armed:
                    reason = 'waiting_for_interception_reset'
                    self._log_error_once(reason)
                    self._publish_uninitialized(reason)
                    return
                if self.initial_q_b_to_e is None:
                    reason = 'waiting_for_initial_vehicle_attitude'
                    self._log_error_once(reason)
                    self._publish_uninitialized(reason)
                    return
                if not message.valid:
                    self._publish_uninitialized(
                        message.reason or 'image_invalid'
                    )
                    return
                snapshot = self.observer.initialize_from_image(
                    self.initial_q_b_to_e,
                    image_xy,
                    self.initial_depth_m,
                    self.initial_v_r_e,
                    self.initial_b_gyr_b,
                    self.initial_b_acc_b,
                )
                self.last_imu_timestamp_us = None
                self.get_logger().info(
                    '18-state observer initialized from attitude, image, and '
                    f'{self.initial_depth_m:.2f} m depth prior'
                )
                self._publish(snapshot, 'initialized')
                return
            self._predict_pending_imu()
            if message.valid:
                snapshot = self.observer.correct_image(image_xy)
                self._publish(snapshot, 'image_corrected_d0')
            else:
                # Keep one observer publication per camera frame, while
                # marking the state unusable for the controller on a frame
                # without a valid target measurement.
                self._publish(
                    self.observer.snapshot(),
                    message.reason or 'image_invalid',
                    valid=False,
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            reason = str(error)
            self._log_error_once(reason)
            if self.observer.initialized:
                self._publish(
                    self.observer.snapshot(),
                    reason,
                    valid=False,
                )
            else:
                self._publish_uninitialized(reason)

    def _imu_callback(self, message: SensorCombined) -> None:
        if not self.observer.initialized:
            return
        timestamp_us = int(message.timestamp)
        if (
            self.last_queued_imu_timestamp_us is not None
            and timestamp_us <= self.last_queued_imu_timestamp_us
        ):
            return
        try:
            gyro_b = frd_to_flu(message.gyro_rad)
            accel_b = frd_to_flu(message.accelerometer_m_s2)
            self.pending_imu_samples.append(
                (timestamp_us, gyro_b, accel_b)
            )
            self.last_queued_imu_timestamp_us = timestamp_us
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self._log_error_once(str(error))

    def _predict_pending_imu(self) -> None:
        """Integrate all IMU samples since the previous image update."""
        while self.pending_imu_samples:
            timestamp_us, gyro_b, accel_b = self.pending_imu_samples.pop(0)
            if self.last_imu_timestamp_us is None:
                self.last_imu_timestamp_us = timestamp_us
                continue
            dt_s = (timestamp_us - self.last_imu_timestamp_us) * 1e-6
            self.last_imu_timestamp_us = timestamp_us
            if dt_s <= 0.0:
                raise ValueError(
                    f'dt_s must be positive, got {dt_s}'
                )
            maximum_dt_s = self.observer.config.maximum_dt_s
            # Keep the observer's Jacobian step bounded, but do not discard a
            # valid IMU interval just because ROS scheduling delivered it late.
            remaining_s = dt_s
            while remaining_s > maximum_dt_s:
                self.observer.predict(
                    gyro_b,
                    accel_b,
                    maximum_dt_s,
                )
                remaining_s -= maximum_dt_s
            if remaining_s > 0.0:
                self.observer.predict(gyro_b, accel_b, remaining_s)

    def _publish(
        self,
        snapshot: ObserverSnapshot,
        reason: str,
        *,
        valid: bool = True,
    ) -> None:
        state = snapshot.state
        message = ObserverState()
        message.stamp = self.get_clock().now().to_msg()
        message.q = state.q
        _assign_vector(message.p_r, state.p_r_e)
        _assign_vector(message.v_r, state.v_r_e)
        message.image_xy = state.image_xy
        _assign_vector(message.b_gyr, state.b_gyr_b)
        _assign_vector(message.b_acc, state.b_acc_b)
        message.covariance = tuple(snapshot.covariance.reshape(-1))
        message.prediction_count = snapshot.prediction_count
        message.correction_count = snapshot.correction_count
        message.innovation_norm = snapshot.innovation_norm
        message.initialized = True
        message.valid = valid
        message.reason = reason
        self.publisher.publish(message)
        if valid:
            self.last_error = ''

    def _publish_uninitialized(self, reason: str) -> None:
        """Publish an invalid status before first initialization."""
        message = ObserverState()
        message.stamp = self.get_clock().now().to_msg()
        message.initialized = False
        message.valid = False
        message.reason = reason
        self.publisher.publish(message)

    def _log_error_once(self, reason: str) -> None:
        if reason != self.last_error:
            self.get_logger().warning(reason)
            self.last_error = reason

    def _positive(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def _vector_parameter(self, name: str, length: int) -> np.ndarray:
        values = np.asarray(self.get_parameter(name).value, dtype=float)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must contain {length} finite values')
        return values


def _assign_vector(message, values) -> None:
    message.x, message.y, message.z = (float(value) for value in values)


def main(args=None) -> None:
    """Run the paper state observer until ROS shutdown."""
    rclpy.init(args=args)
    node = PaperStateObserverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
