"""Detect the red Gazebo target and publish an undelayed image feature."""

from typing import Optional

from interception_interfaces.msg import VisionFeature
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

from ibvs_perception.camera_geometry import (
    CameraIntrinsics,
    intrinsics_from_horizontal_fov,
    normalized_pixel,
    red_hsv_mask,
)


class RedTargetDetector(Node):
    """Run a deterministic red HSV segmentation at the camera arrival rate."""

    def __init__(self) -> None:
        super().__init__('red_target_detector')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('feature_topic', '/interception/vision/raw_feature')
        self.declare_parameter('fallback_width', 1280)
        self.declare_parameter('fallback_height', 960)
        self.declare_parameter('fallback_horizontal_fov_rad', 1.74)
        self.declare_parameter('minimum_area_px', 20.0)
        self.declare_parameter('detector_stride', 2)
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.sequence = 0
        # A visual servo must prefer the newest frame over processing a FIFO
        # backlog.  This is not a delay model: old frames are discarded.
        latest_sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(
            VisionFeature,
            str(self.get_parameter('feature_topic').value),
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self._camera_info_callback,
            latest_sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self._image_callback,
            latest_sensor_qos,
        )
        self.get_logger().info(
            'Red target detector ready: direct camera arrival, no delay queue'
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        try:
            candidate = CameraIntrinsics(
                width=int(message.width),
                height=int(message.height),
                fx=float(message.k[0]),
                fy=float(message.k[4]),
                cx=float(message.k[2]),
                cy=float(message.k[5]),
            )
            candidate.validate()
            self.intrinsics = candidate
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f'Ignoring invalid CameraInfo: {error}')

    def _image_callback(self, message: Image) -> None:
        feature = VisionFeature()
        feature.capture_stamp = message.header.stamp
        feature.publish_stamp = self.get_clock().now().to_msg()
        feature.sequence = self.sequence
        self.sequence += 1
        try:
            rgb = _image_to_rgb(message)
            intrinsics = self._intrinsics_for_image(message)
            stride = int(self.get_parameter('detector_stride').value)
            if stride <= 0:
                raise ValueError('detector_stride must be positive')
            # Gazebo's bundled camera is 1280x960.  Segmenting every second
            # sample implements the planned 640x480 detector resolution while
            # retaining raw-image pixel coordinates for camera geometry.
            mask = red_hsv_mask(rgb[::stride, ::stride])
            area = float(np.count_nonzero(mask) * stride * stride)
            feature.area_px = area
            if area < float(self.get_parameter('minimum_area_px').value):
                feature.valid = False
                feature.reason = 'red_target_not_found'
            else:
                rows, columns = np.nonzero(mask)
                pixel_offset = (stride - 1) / 2.0
                feature.u = float(np.mean(columns) * stride + pixel_offset)
                feature.v = float(np.mean(rows) * stride + pixel_offset)
                feature.x_norm, feature.y_norm = normalized_pixel(
                    feature.u,
                    feature.v,
                    intrinsics,
                )
                feature.valid = True
                feature.reason = 'ok'
        except ValueError as error:
            feature.valid = False
            feature.reason = str(error)
        self.publisher.publish(feature)

    def _intrinsics_for_image(self, image: Image) -> CameraIntrinsics:
        intrinsics = self.intrinsics
        if (
            intrinsics is not None
            and intrinsics.width == image.width
            and intrinsics.height == image.height
        ):
            return intrinsics
        width = int(image.width) or int(self.get_parameter('fallback_width').value)
        height = int(image.height) or int(self.get_parameter('fallback_height').value)
        return intrinsics_from_horizontal_fov(
            width,
            height,
            float(self.get_parameter('fallback_horizontal_fov_rad').value),
        )


def _image_to_rgb(message: Image) -> np.ndarray:
    """Decode common uncompressed bridge image encodings into RGB pixels."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        'rgb8': (3, (0, 1, 2)),
        'bgr8': (3, (2, 1, 0)),
        'rgba8': (4, (0, 1, 2)),
        'bgra8': (4, (2, 1, 0)),
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f'unsupported image encoding: {message.encoding}')
    channels, order = channels_by_encoding[encoding]
    if message.height <= 0 or message.width <= 0:
        raise ValueError('image dimensions must be positive')
    minimum_step = message.width * channels
    if message.step < minimum_step or len(message.data) < message.step * message.height:
        raise ValueError('image data does not match dimensions and step')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        (message.height, message.step)
    )
    pixels = rows[:, :minimum_step].reshape((message.height, message.width, channels))
    return pixels[:, :, order]


def main(args=None) -> None:
    """Run the undelayed red target detector."""
    rclpy.init(args=args)
    node = RedTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
