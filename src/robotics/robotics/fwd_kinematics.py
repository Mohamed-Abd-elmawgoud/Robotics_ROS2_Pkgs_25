#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped, Pose
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R


class ForwardKinematicsNode(Node):
    def __init__(self):
        super().__init__('forward_kinematics_node')

        # Link lengths (mm)
        self.l0 = 50.0
        self.l1 = 43.55
        self.l2 = 139.79
        self.l3 = 104.07
        self.l4 = 30.01
        self.l5 = 69.35

        # Initialize joint angles
        self.q = [0.0] * 6

        # Subscribers and publishers
        self.joint_sub = self.create_subscription(
            Float64MultiArray,
            'joint_angles',
            self.joint_angles_callback,
            10
        )

        self.pose_pub = self.create_publisher(Pose, 'end_effector_pose', 10)
        self.joint_angles_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)
        self.position_pub = self.create_publisher(Float64MultiArray, 'end_effector_position', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("Forward Kinematics Node started (EE offset removed).")

    # ------------------------------------------------------------
    # Forward kinematics using DH parameters
    # ------------------------------------------------------------
    def fwd_kinematics(self, q):
        l0, l1, l2, l3, l4, l5 = self.l0, self.l1, self.l2, self.l3, self.l4, self.l5

        theta = [
            np.pi/2 + q[0],
            0,
            np.pi/2 + q[1],
            -q[2] - np.pi/2,
            -q[3],
            np.pi/2 + q[4]
        ]
        d = [l0, l1, 0, 0, l3 + l4, 0]
        a = [0, 0, l2, 0, 0, l5]
        alpha = [0, np.pi/2, 0, -np.pi/2, np.pi/2, 0]

        T = np.eye(4)
        for i in range(6):
            ct, st = np.cos(theta[i]), np.sin(theta[i])
            ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
            T_i = np.array([
                [ct, -st * ca, st * sa, a[i] * ct],
                [st, ct * ca, -ct * sa, a[i] * st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
        return T

    # ------------------------------------------------------------
    # Apply FK and publish EE pose
    # ------------------------------------------------------------
    def compute_and_publish_fk(self):
        # Publish joint angles for other nodes
        joint_msg = Float64MultiArray()
        joint_msg.data = self.q
        self.joint_angles_pub.publish(joint_msg)

        # Compute FK to end effector (no offset applied)
        T = self.fwd_kinematics(self.q)

        # Extract position and rotation
        pos = T[:3, 3] / 1000.0  # convert from mm to meters
        rot = R.from_matrix(T[:3, :3])
        quat = rot.as_quat()  # [x, y, z, w]

        # Publish EE position
        pos_msg = Float64MultiArray()
        pos_msg.data = pos.tolist()
        self.position_pub.publish(pos_msg)

        # Publish EE pose
        pose_msg = Pose()
        pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = pos
        pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = quat
        self.pose_pub.publish(pose_msg)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'Base_Link'
        t.child_frame_id = 'EE_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------
    # Callback for joint angles
    # ------------------------------------------------------------
    def joint_angles_callback(self, msg):
        if len(msg.data) != 6:
            self.get_logger().warn(f"Expected 6 joint angles, got {len(msg.data)}")
            return
        self.q = list(msg.data)
        self.compute_and_publish_fk()


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
