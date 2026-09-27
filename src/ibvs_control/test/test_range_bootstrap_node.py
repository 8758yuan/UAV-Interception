"""ROS callback tests for range-bootstrap observer initialization."""

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from interception_interfaces.msg import ObserverBootstrap, VisionFeature
from px4_msgs.msg import VehicleAttitude
import rclpy
from std_msgs.msg import Empty

from ibvs_control.paper_state_observer_node import PaperStateObserverNode


@pytest.fixture
def observer_node():
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init()
    node = None
    try:
        node = PaperStateObserverNode()
        assert node.require_range_bootstrap
        attitude = VehicleAttitude()
        attitude.q = [1.0, 0.0, 0.0, 0.0]
        node._attitude_callback(attitude)
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def _stamp(nanoseconds: int) -> Time:
    stamp = Time()
    stamp.sec = int(nanoseconds // 1_000_000_000)
    stamp.nanosec = int(nanoseconds % 1_000_000_000)
    return stamp


def _stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _feature(capture_ns: int) -> VisionFeature:
    message = VisionFeature()
    message.capture_stamp = _stamp(capture_ns)
    message.valid = True
    message.x_norm = 0.02
    message.y_norm = -0.01
    return message


def _bootstrap(
    node: PaperStateObserverNode,
    *,
    age_ns: int = 50_000_000,
) -> tuple[ObserverBootstrap, np.ndarray]:
    message = ObserverBootstrap()
    message.stamp = _stamp(node.get_clock().now().nanoseconds - age_ns)
    message.p_r.x = -10.0
    message.p_r.y = 1.5
    message.p_r.z = -0.4
    message.v_r.x = 0.4
    message.v_r.y = -0.2
    message.v_r.z = 0.1
    covariance = np.diag((4.0, 2.0, 1.0, 0.25, 0.16, 0.09))
    covariance[0, 3] = covariance[3, 0] = 0.4
    message.relative_covariance = covariance.reshape(-1).tolist()
    message.sample_count = 20
    message.range_m = 10.1
    message.range_std_m = 0.7
    return message, covariance


def _assert_initialized_from_bootstrap(
    node: PaperStateObserverNode,
    bootstrap: ObserverBootstrap,
    covariance: np.ndarray,
    capture_ns: int,
) -> None:
    snapshot = node.observer.snapshot()
    dt_s = (capture_ns - _stamp_ns(bootstrap.stamp)) * 1e-9
    position = np.array(
        (bootstrap.p_r.x, bootstrap.p_r.y, bootstrap.p_r.z)
    )
    velocity = np.array(
        (bootstrap.v_r.x, bootstrap.v_r.y, bootstrap.v_r.z)
    )
    assert np.asarray(snapshot.state.p_r_e) == pytest.approx(
        position + velocity * dt_s
    )
    assert snapshot.state.v_r_e == pytest.approx(velocity)
    assert snapshot.state.image_xy == pytest.approx((0.02, -0.01))
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt_s
    assert snapshot.covariance[4:10, 4:10] == pytest.approx(
        transition @ covariance @ transition.T
    )
    assert snapshot.prediction_count == 0
    assert node.pending_bootstrap is None


def test_reset_and_valid_image_wait_for_range_bootstrap(observer_node) -> None:
    node = observer_node
    node._reset_callback(Empty())
    node._feature_callback(_feature(node.get_clock().now().nanoseconds))
    assert node.initialization_armed
    assert not node.observer.initialized
    assert node.image_update_count == 1


@pytest.mark.parametrize('bootstrap_before_reset', (False, True))
def test_fresh_image_initializes_bootstrap_state_in_either_message_order(
    observer_node,
    bootstrap_before_reset: bool,
) -> None:
    node = observer_node
    bootstrap, covariance = _bootstrap(node)
    if bootstrap_before_reset:
        node._bootstrap_callback(bootstrap)
        node._reset_callback(Empty())
    else:
        node._reset_callback(Empty())
        node._bootstrap_callback(bootstrap)
    assert node.pending_bootstrap is bootstrap
    capture_ns = node.get_clock().now().nanoseconds
    node._feature_callback(_feature(capture_ns))
    assert node.observer.initialized
    _assert_initialized_from_bootstrap(
        node, bootstrap, covariance, capture_ns
    )


def test_stale_bootstrap_is_rejected(observer_node) -> None:
    node = observer_node
    node._reset_callback(Empty())
    stale, _ = _bootstrap(
        node,
        age_ns=int((node.bootstrap_max_age_s + 1.0) * 1e9),
    )
    node._bootstrap_callback(stale)
    assert node.pending_bootstrap is None
    node._feature_callback(_feature(node.get_clock().now().nanoseconds))
    assert not node.observer.initialized

    fresh, covariance = _bootstrap(node)
    node._bootstrap_callback(fresh)
    capture_ns = node.get_clock().now().nanoseconds
    node._feature_callback(_feature(capture_ns))
    assert node.observer.initialized
    _assert_initialized_from_bootstrap(node, fresh, covariance, capture_ns)


def test_image_captured_before_reset_is_rejected(observer_node) -> None:
    node = observer_node
    node._reset_callback(Empty())
    bootstrap, covariance = _bootstrap(node)
    node._bootstrap_callback(bootstrap)
    assert node.pending_bootstrap is bootstrap
    node._feature_callback(_feature(node.reset_requested_ns - 1))
    assert not node.observer.initialized
    assert node.pending_bootstrap is bootstrap

    capture_ns = node.get_clock().now().nanoseconds
    node._feature_callback(_feature(capture_ns))
    assert node.observer.initialized
    _assert_initialized_from_bootstrap(
        node, bootstrap, covariance, capture_ns
    )
