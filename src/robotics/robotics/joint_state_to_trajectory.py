#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np

class JointPositionWithEEComparison(Node):
    def __init__(self):
        super().__init__('joint_position_with_ee_comparison')

        # Publisher for joint positions (forward_command_controller)
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Subscriptions
        self.joint_angles_sub = self.create_subscription(
            Float64MultiArray,
            '/joint_angles_out',
            self.joint_angles_callback,
            10
        )

        self.ee_fk_sub = self.create_subscription(
            Float64MultiArray,
            '/end_effector_position',
            self.ee_fk_callback,
            10
        )

        self.ee_gazebo_sub = self.create_subscription(
            PoseStamped,
            '/ee_pose',
            self.ee_gazebo_callback,
            10
        )

        # Publisher for EE position error
        self.error_pub = self.create_publisher(Float64MultiArray, '/ee_position_error', 10)

        # Internal state
        self.last_joint_angles = None
        self.ee_fk_pos = None
        self.ee_gz_pos = None

        self.get_logger().info(
            'Joint Position + EE Comparison Node started.\n'
            'Publishing to /joint_group_position_controller/commands\n'
            'Subscribed to /joint_angles_out, /end_effector_position, /ee_pose\n'
            'Publishing EE error on /ee_position_error'
        )

    # Publish joint positions to forward_command_controller
    def joint_angles_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 5:
            self.get_logger().warn(f"Received insufficient joint angles: {len(msg.data)}")
            return

        positions = msg.data[:5]  # Use first 5 joints
        if self.last_joint_angles is None or positions != self.last_joint_angles:
            self.last_joint_angles = positions

            cmd_msg = Float64MultiArray()
            cmd_msg.data = positions
            self.joint_cmd_pub.publish(cmd_msg)

            self.get_logger().info(f"Published joint positions: {np.round(positions, 4)}")

    # Handle FK-computed EE position
    def ee_fk_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 3:
            self.get_logger().warn("Invalid FK EE position length")
            return
        self.ee_fk_pos = np.array(msg.data)
        self.compare_positions()

    # Handle Gazebo EE pose
    def ee_gazebo_callback(self, msg: PoseStamped):
        self.ee_gz_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.compare_positions()

    # Compare FK EE and Gazebo EE positions
    def compare_positions(self):
        if self.ee_fk_pos is not None and self.ee_gz_pos is not None:
            diff = self.ee_fk_pos - self.ee_gz_pos
            error = np.linalg.norm(diff)

            msg = Float64MultiArray()
            msg.data = [float(diff[0]), float(diff[1]), float(diff[2]), float(error)]
            self.error_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointPositionWithEEComparison()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
