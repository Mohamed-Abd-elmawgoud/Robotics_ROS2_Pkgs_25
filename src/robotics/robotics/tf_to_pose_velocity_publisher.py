#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros


class TFtoPosePublisher(Node):
    def __init__(self):
        super().__init__('tf_to_pose_velocity_publisher')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10 Hz

        self.parent_frame = 'base_link'
        self.child_frame = 'EE_frame'
        self.get_logger().info(f'Publishing /ee_pose from TF [{self.parent_frame} → {self.child_frame}]')

    def publish_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time()
            )

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.parent_frame
            msg.pose.position.x = trans.transform.translation.x
            msg.pose.position.y = trans.transform.translation.y
            msg.pose.position.z = trans.transform.translation.z
            msg.pose.orientation = trans.transform.rotation

            self.pose_pub.publish(msg)
        except Exception as e:
            # This will happen briefly before TF tree is ready
            self.get_logger().debug(f"TF lookup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TFtoPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()