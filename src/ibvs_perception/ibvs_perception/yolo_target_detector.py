"""Publish the existing VisionFeature interface from YOLO bounding boxes."""

from collections import deque
import math
from pathlib import Path
import time

from cv_bridge import CvBridge, CvBridgeError
import cv2
from interception_interfaces.msg import VisionFeature
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty

from ibvs_perception.camera_geometry import (
    CameraIntrinsics,
    PAPER_HORIZONTAL_FOV_RAD,
    intrinsics_from_horizontal_fov,
    normalized_pixel,
)
from ibvs_perception.yolo_detection import (
    delayed_release_ns,
    select_target_box,
)


def _stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class YoloTargetDetector(Node):
    """Run one configured YOLO detector on the existing Gazebo camera topic."""

    def __init__(self) -> None:
        super().__init__('yolo_target_detector')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter(
            'feature_topic', '/interception/vision/raw_feature'
        )
        self.declare_parameter(
            'debug_image_topic', '/interception/vision/debug_image'
        )
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_class', 'sports ball')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('inference_size', 640)
        self.declare_parameter('device', '')
        self.declare_parameter('image_delay_s', 0.08)
        self.declare_parameter('delay_poll_period_s', 0.005)
        self.declare_parameter('fallback_width', 1280)
        self.declare_parameter('fallback_height', 960)
        self.declare_parameter(
            'fallback_horizontal_fov_rad', PAPER_HORIZONTAL_FOV_RAD
        )
        self.declare_parameter(
            'observer_reset_topic', '/interception/observer/reset'
        )

        model_path = Path(
            str(self.get_parameter('model_path').value)
        ).expanduser()
        if not model_path.is_file():
            raise ValueError(
                f'model_path must name an existing YOLO weight file: '
                f'{model_path}'
            )
        self.target_class = str(
            self.get_parameter('target_class').value
        ).strip()
        if not self.target_class:
            raise ValueError(
                'target_class must name one class in the YOLO model'
            )
        self.confidence_threshold = self._unit_interval_parameter(
            'confidence_threshold'
        )
        self.iou_threshold = self._unit_interval_parameter('iou_threshold')
        self.inference_size = int(self.get_parameter('inference_size').value)
        if self.inference_size <= 0:
            raise ValueError('inference_size must be positive')
        self.device = str(self.get_parameter('device').value).strip()
        self.image_delay_s = self._nonnegative_parameter('image_delay_s')
        poll_period_s = self._positive_parameter('delay_poll_period_s')

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if self.model.task != 'detect':
            raise ValueError('model_path must contain YOLO detection weights')
        names = self.model.names
        self.class_names = (
            dict(enumerate(names))
            if isinstance(names, (list, tuple))
            else dict(names)
        )
        matching_ids = [
            int(class_id)
            for class_id, name in self.class_names.items()
            if str(name).casefold() == self.target_class.casefold()
        ]
        if not matching_ids:
            raise ValueError(
                f'target_class {self.target_class!r} is absent from '
                f'{model_path}; '
                f'available classes: {list(self.class_names.values())}'
            )
        self.target_class_id = min(matching_ids)
        self.bridge = CvBridge()
        self.intrinsics = None
        self.sequence = 0
        self.delayed_features = deque()
        self.last_frame_finished_s = None
        self.last_error = ''

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.feature_pub = self.create_publisher(
            VisionFeature, str(self.get_parameter('feature_topic').value), 10
        )
        self.debug_pub = None
        if bool(self.get_parameter('publish_debug_image').value):
            self.debug_pub = self.create_publisher(
                Image,
                str(self.get_parameter('debug_image_topic').value),
                sensor_qos,
            )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self._camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self._image_callback,
            sensor_qos,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter('observer_reset_topic').value),
            self._reset_callback,
            10,
        )
        self.create_timer(poll_period_s, self._publish_ready_features)
        self.get_logger().info(
            f'YOLO ready: {model_path}, class={self.target_class!r}, '
            f'image delay={self.image_delay_s * 1000:.0f} ms'
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

    def _intrinsics_for_image(self, image: Image) -> CameraIntrinsics:
        intrinsics = self.intrinsics
        if (
            intrinsics is not None
            and intrinsics.width == image.width
            and intrinsics.height == image.height
        ):
            return intrinsics
        width = int(image.width) or int(
            self.get_parameter('fallback_width').value
        )
        height = int(image.height) or int(
            self.get_parameter('fallback_height').value
        )
        return intrinsics_from_horizontal_fov(
            width,
            height,
            float(self.get_parameter('fallback_horizontal_fov_rad').value),
        )

    def _image_callback(self, message: Image) -> None:
        started_s = time.perf_counter()
        feature = VisionFeature()
        feature.capture_stamp = message.header.stamp
        feature.sequence = self.sequence
        self.sequence += 1
        # A lost target has no valid pixel coordinates; downstream uses valid.
        feature.u = math.nan
        feature.v = math.nan
        feature.x_norm = math.nan
        feature.y_norm = math.nan
        feature.area_px = 0.0
        feature.valid = False
        feature.reason = 'target_class_not_detected'
        frame = None
        selected = None
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            intrinsics = self._intrinsics_for_image(message)
            options = {
                'source': frame,
                'imgsz': self.inference_size,
                'conf': self.confidence_threshold,
                'iou': self.iou_threshold,
                'verbose': False,
            }
            if self.device:
                options['device'] = self.device
            result = self.model.predict(**options)[0]
            boxes = result.boxes
            if boxes is not None:
                selected = select_target_box(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    target_class_id=self.target_class_id,
                    image_width=message.width,
                    image_height=message.height,
                )
            if selected is not None:
                feature.u, feature.v = selected.center
                feature.x_norm, feature.y_norm = normalized_pixel(
                    feature.u, feature.v, intrinsics
                )
                feature.area_px = selected.area_px
                feature.valid = True
                feature.reason = 'ok'
            self.last_error = ''
        except (
            CvBridgeError,
            RuntimeError,
            TypeError,
            ValueError,
            IndexError,
        ) as error:
            feature.reason = f'yolo_detection_error: {error}'
            if feature.reason != self.last_error:
                self.get_logger().error(feature.reason)
                self.last_error = feature.reason

        finished_s = time.perf_counter()
        interval_s = (
            finished_s - self.last_frame_finished_s
            if self.last_frame_finished_s is not None
            else finished_s - started_s
        )
        self.last_frame_finished_s = finished_s
        fps = 1.0 / interval_s if interval_s > 0.0 else 0.0
        now_ns = self.get_clock().now().nanoseconds
        release_ns = delayed_release_ns(
            _stamp_to_nanoseconds(feature.capture_stamp),
            now_ns,
            self.image_delay_s,
        )
        self.delayed_features.append((release_ns, feature))
        self._publish_ready_features()
        if (
            self.debug_pub is not None
            and self.debug_pub.get_subscription_count() > 0
            and frame is not None
        ):
            try:
                self._publish_debug_image(frame, message, selected, fps)
            except (CvBridgeError, RuntimeError, ValueError) as error:
                self.get_logger().warning(f'YOLO debug image skipped: {error}')

    def _publish_debug_image(
        self, frame: object, source: Image, selected, fps: float
    ) -> None:
        annotated = frame.copy()
        if selected is not None:
            x1, y1, x2, y2 = (
                int(round(selected.x1)),
                int(round(selected.y1)),
                int(round(selected.x2)),
                int(round(selected.y2)),
            )
            u, v = selected.center
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(
                annotated,
                (int(round(u)), int(round(v))),
                5,
                (0, 0, 255),
                -1,
            )
            caption = f'{self.target_class} {selected.confidence:.2f}'
            cv2.putText(
                annotated,
                caption,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                annotated,
                f'{self.target_class}: not detected',
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            annotated,
            f'FPS {fps:.1f}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        debug = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug.header = source.header
        self.debug_pub.publish(debug)

    def _publish_ready_features(self) -> None:
        now = self.get_clock().now()
        while (
            self.delayed_features
            and self.delayed_features[0][0] <= now.nanoseconds
        ):
            _, feature = self.delayed_features.popleft()
            feature.publish_stamp = now.to_msg()
            self.feature_pub.publish(feature)

    def _reset_callback(self, _message: Empty) -> None:
        self.delayed_features.clear()

    def _nonnegative_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and nonnegative')
        return value

    def _positive_parameter(self, name: str) -> float:
        value = self._nonnegative_parameter(name)
        if value == 0.0:
            raise ValueError(f'{name} must be positive')
        return value

    def _unit_interval_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f'{name} must be in (0, 1]')
        return value


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTargetDetector()
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
