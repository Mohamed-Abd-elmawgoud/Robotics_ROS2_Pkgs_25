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

        # Link lengths (mm) - Updated to match MATLAB code
        self.l1 = 41.05
        self.l2 = 139.93
        self.l3 = 132.9
        self.l4 = 52.3
        self.l5 = 37.76

        # Initialize joint angles (now 5 joints instead of 6)
        self.q = [0.0] * 5

        # Subscribers and publishers
        self.joint_sub = self.create_subscription(
            Float64MultiArray,
            'joint_angles',
            self.joint_angles_callback,
            10
        )

        self.pose_pub = self.create_publisher(Pose, 'end_effector_pose', 10)
        self.position_pub = self.create_publisher(Float64MultiArray, 'end_effector_position', 10)
        self.joint_angles_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("Forward Kinematics Node started (5-DOF, MATLAB converted).")

    # ------------------------------------------------------------
    # Forward kinematics using DH parameters (converted from MATLAB)
    # ------------------------------------------------------------
    def fwd_kinematics(self, q):
        """
        Computes the overall transformation matrix from DH parameters
        
        Input:
            q - list of 5 joint angles [q1, q2, q3, q4, q5]
        Output:
            T - 4x4 homogeneous transformation matrix of the end effector
        """
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        
        # DH parameter table (converted from MATLAB)
        theta = [
            np.pi/2 + q[0],
            -np.pi/2 + q[1],
            q[2],
            -np.pi/2 + q[3],
            np.pi/2 + q[4]
        ]
        d = [l1, 0, 0, 0, -l4-l5]
        a = [0, l2, l3, 0, 0]
        alpha = [-np.pi/2, np.pi, np.pi, np.pi/2, 0]
        
        # Initialize transformation matrix
        T = np.eye(4)
        
        # Compute transformation for each joint
        for i in range(5):
            ct = np.cos(theta[i])
            st = np.sin(theta[i])
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])
            
            # Transformation from frame i-1 to frame i
            T_i = np.array([
                [ct, -st * ca,  st * sa, a[i] * ct],
                [st,  ct * ca, -ct * sa, a[i] * st],
                [0,   sa,       ca,      d[i]],
                [0,   0,        0,       1]
            ])
            
            # Multiply cumulative transformation
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

        # Compute FK to end effector
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
        if len(msg.data) != 5:
            self.get_logger().warn(f"Expected 5 joint angles, got {len(msg.data)}")
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