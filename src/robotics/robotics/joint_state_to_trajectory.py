#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
import numpy as np


class JointTrajectoryWithEEComparison(Node):
    def __init__(self):
        super().__init__('joint_trajectory_with_ee_comparison')

        # --- ROS 2 action client for sending joint trajectory goals ---
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # --- Subscriptions ---
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

        # --- Publisher for EE position error ---
        self.error_pub = self.create_publisher(Float64MultiArray, '/ee_position_error', 10)

        # --- Internal state ---
        self.last_joint_angles = None
        self.ee_fk_pos = None
        self.ee_gz_pos = None

        self.get_logger().info(
            'Joint Trajectory + EE Comparison Node started.\n'
            'Subscribed to:\n'
            '  • /joint_angles_out\n'
            '  • /end_effector_position\n'
            '  • /ee_pose\n'
            'Publishing errors on:\n'
            '  • /ee_position_error'
        )

    # ============================================================
    # (1) Handle Joint Angles -> Send as JointTrajectory goal
    # ============================================================
    def joint_angles_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 5:
            self.get_logger().warn(f"Received insufficient joint angles: {len(msg.data)}")
            return

        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5']
        positions = msg.data[:5]  # Only first 5 joints if 6 are sent

        if self.last_joint_angles is None or positions != self.last_joint_angles:
            self.last_joint_angles = positions

            traj = JointTrajectory()
            traj.header.frame_id = 'world'
            traj.joint_names = joint_names

            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * len(joint_names)
            point.time_from_start = Duration(sec=3)

            traj.points.append(point)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            self.trajectory_client.send_goal_async(goal)

            self.get_logger().info(f"Sent joint trajectory: {np.round(positions, 4)}")

    # ============================================================
    # (2) Handle FK-computed EE position
    # ============================================================
    def ee_fk_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 3:
            self.get_logger().warn("Invalid FK EE position length")
            return
        self.ee_fk_pos = np.array(msg.data)
        self.compare_positions()

    # ============================================================
    # (3) Handle Gazebo EE pose
    # ============================================================
    def ee_gazebo_callback(self, msg: PoseStamped):
        self.ee_gz_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.compare_positions()

    # ============================================================
    # (4) Compare FK EE and Gazebo EE positions (no offset applied)
    # ============================================================
    def compare_positions(self):
        if self.ee_fk_pos is not None and self.ee_gz_pos is not None:
            diff = self.ee_fk_pos - self.ee_gz_pos
            error = np.linalg.norm(diff)

            msg = Float64MultiArray()
            msg.data = [float(diff[0]), float(diff[1]), float(diff[2]), float(error)]
            self.error_pub.publish(msg)

            if error > 0.01:
                self.get_logger().warn(
                    f"[EE Error] Δx={diff[0]:.4f}, Δy={diff[1]:.4f}, Δz={diff[2]:.4f} |Δ|={error:.4f} m"
                )


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryWithEEComparison()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
