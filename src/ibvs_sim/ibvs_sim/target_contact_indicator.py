"""Change the Gazebo target from red to green after a verified contact."""

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
from gz.msgs10.visual_pb2 import Visual
from gz.transport13 import Node as GazeboNode
from google.protobuf.message import DecodeError

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool


class TargetContactIndicator(Node):
    """Apply a one-shot green material update on real target contact."""

    def __init__(self) -> None:
        super().__init__('target_contact_indicator')
        self.declare_parameter(
            'contact_topic',
            '/world/default/model/ibvs_target/link/target_link/sensor/'
            'target_contact/contact',
        )
        self.declare_parameter(
            'visual_name',
            'target_visual',
        )
        self.declare_parameter('visual_parent_name', 'target_link')
        self.declare_parameter('visual_config_service', '/world/default/visual_config')
        self.declare_parameter('world_state_service', '/world/default/state')
        self.declare_parameter(
            'target_collision_name',
            'ibvs_target::target_link::target_collision',
        )
        self.declare_parameter(
            'vehicle_collision_prefix',
            'x500_mono_cam_0::',
        )
        self.declare_parameter(
            'confirmation_topic',
            '/interception/target/green_confirmed',
        )
        self.changed = False
        self.visual_update_pending = False
        self.verification_attempts = 0
        self.gz_node = GazeboNode()
        confirmation_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.confirmation_pub = self.create_publisher(
            Bool,
            str(self.get_parameter('confirmation_topic').value),
            confirmation_qos,
        )
        self.create_subscription(
            Contacts,
            str(self.get_parameter('contact_topic').value),
            self._contact_callback,
            qos_profile_sensor_data,
        )
        self.verification_timer = self.create_timer(
            0.1,
            self._verify_green_material,
        )
        self.get_logger().info('Target contact indicator ready; target is red')

    def _contact_callback(self, message: Contacts) -> None:
        if self.changed or not _has_expected_target_contact(
            message,
            str(self.get_parameter('target_collision_name').value),
            str(self.get_parameter('vehicle_collision_prefix').value),
        ):
            return
        visual_name = str(self.get_parameter('visual_name').value)
        parent_name = str(
            self.get_parameter('visual_parent_name').value
        )
        service = str(self.get_parameter('visual_config_service').value)
        request = _green_visual_message(visual_name, parent_name)
        executed, response = self.gz_node.request(
            service,
            request,
            Visual,
            Boolean,
            2000,
        )
        if not executed or not response.data:
            self.get_logger().error(
                'Gazebo rejected the target material update request'
            )
            return
        # The visual_config response only means that the command was queued.
        # Confirm the applied VisualCmd in Gazebo's world state before
        # announcing a successful red-to-green transition.
        self.visual_update_pending = True
        self.verification_attempts = 0
        # VisualCmd is a one-shot component which renderers may consume on the
        # next update, so inspect the full state immediately.  The timer below
        # remains as a short retry path if the state service is momentarily
        # unavailable.
        self._verify_green_material()

    def _verify_green_material(self) -> None:
        if self.changed or not self.visual_update_pending:
            return
        self.verification_attempts += 1
        service = str(self.get_parameter('world_state_service').value)
        executed, state = self.gz_node.request(
            service,
            Empty(),
            Empty,
            SerializedStepMap,
            2000,
        )
        visual_name = str(self.get_parameter('visual_name').value)
        parent_name = str(
            self.get_parameter('visual_parent_name').value
        )
        if not executed or not _state_has_green_visual_command(
            state,
            visual_name,
            parent_name,
        ):
            if self.verification_attempts >= 20:
                self.visual_update_pending = False
                self.get_logger().error(
                    'Target contact occurred, but Gazebo did not confirm the '
                    'green target material'
                )
            return
        self.changed = True
        self.visual_update_pending = False
        confirmation = Bool()
        confirmation.data = True
        self.confirmation_pub.publish(confirmation)
        self.get_logger().warning('Target contact verified: target changed to green')


def _green_visual_message(
    visual_name: str,
    parent_name: str,
) -> Visual:
    """Build a visual command with the parent required for name lookup."""
    if not visual_name or not parent_name:
        raise ValueError('visual and parent names must not be empty')
    visual = Visual()
    visual.name = visual_name
    visual.parent_name = parent_name
    for color in (visual.material.ambient, visual.material.diffuse):
        color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
    specular = visual.material.specular
    specular.r, specular.g, specular.b, specular.a = 0.1, 1.0, 0.1, 1.0
    emissive = visual.material.emissive
    emissive.r, emissive.g, emissive.b, emissive.a = 0.0, 0.2, 0.0, 1.0
    return visual


def _state_has_green_visual_command(
    state: SerializedStepMap,
    visual_name: str,
    parent_name: str,
) -> bool:
    """Verify Gazebo applied the green VisualCmd to its world state."""
    visual_token = visual_name.encode('utf-8')
    parent_token = parent_name.encode('utf-8')
    for entity in state.state.entities.values():
        for component in entity.components.values():
            payload = component.component
            if (
                not payload.startswith(b'\x12')
                or visual_token not in payload
                or parent_token not in payload
            ):
                continue
            visual = Visual()
            try:
                visual.ParseFromString(payload)
            except DecodeError:
                continue
            if not _name_matches(visual.name, visual_name):
                continue
            if not _name_matches(visual.parent_name, parent_name):
                continue
            diffuse = visual.material.diffuse
            ambient = visual.material.ambient
            if all(
                color.g >= 0.8 and color.r <= 0.2 and color.b <= 0.2
                for color in (diffuse, ambient)
            ):
                return True
    return False


def _name_matches(scoped_name: str, requested_name: str) -> bool:
    return (
        scoped_name == requested_name
        or scoped_name.endswith('::' + requested_name)
    )


def _has_expected_target_contact(
    message: Contacts,
    target_collision_name: str,
    vehicle_collision_prefix: str,
) -> bool:
    """Reject unrelated contacts even if they appear on the sensor topic."""
    for contact in message.contacts:
        names = (contact.collision1.name, contact.collision2.name)
        if target_collision_name not in names:
            continue
        other = names[1] if names[0] == target_collision_name else names[0]
        if other.startswith(vehicle_collision_prefix):
            return True
    return False


def main(args=None) -> None:
    """Run the target-contact visual indicator."""
    rclpy.init(args=args)
    node = TargetContactIndicator()
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
