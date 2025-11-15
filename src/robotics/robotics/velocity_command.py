#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient


class VelocityToTrajectoryNode(Node):
    def __init__(self):
        super().__init__("velocity_to_trajectory_node")

        # Joint names
        self.joint_names = [f"Joint_{i}" for i in range(1, 6)]

        # Current joint state
        self.current_positions = [0.0] * 5
        self.joint_state_received = False

        # Subscribers
        self.create_subscription(
            Float64MultiArray,
            "/desired_velocities",
            self.velocity_callback,
            10
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        # Action client for JointTrajectoryController
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory"
        )

        self.get_logger().info("Velocity → Trajectory node initialized (time inside message).")

    # ----------------------------------------------------------
    # Read joint states
    # ----------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        try:
            indices = [msg.name.index(j) for j in self.joint_names]
            self.current_positions = [msg.position[i] for i in indices]
            self.joint_state_received = True
        except ValueError:
            # JointState not containing required joints yet
            return

    # ----------------------------------------------------------
    # Receive velocity + time_from_start
    # ----------------------------------------------------------
    def velocity_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 6:
            self.get_logger().error("Expected 6 values: 5 joint velocities + time_from_start")
            return

        if not self.joint_state_received:
            self.get_logger().warn("No joint states received yet.")
            return

        # First 5 values = joint velocities
        velocities = msg.data[:5]

        # Last value = time_from_start
        time_from_start = msg.data[5]

        if time_from_start <= 0.0:
            self.get_logger().error("time_from_start must be > 0")
            return

        # ----------------------------------------------------------
        # Integrate velocities to compute final positions
        # ----------------------------------------------------------
        final_positions = [
            self.current_positions[i] + velocities[i] * time_from_start
            for i in range(5)
        ]

        self.get_logger().info(f"Final positions: {final_positions}")

        # Send trajectory goal
        self.send_trajectory(final_positions, time_from_start)

    # ----------------------------------------------------------
    # Build and send the FollowJointTrajectory action goal
    # ----------------------------------------------------------
    def send_trajectory(self, final_positions, duration):
        goal_msg = FollowJointTrajectory.Goal()

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = final_positions
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        traj.points.append(pt)
        goal_msg.trajectory = traj

        self.get_logger().info("Sending trajectory goal...")

        self.action_client.wait_for_server()

        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    # ----------------------------------------------------------
    # Action callbacks
    # ----------------------------------------------------------
    def feedback_callback(self, feedback):
        pass  # ignore continuous feedback

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected!")
            return

        self.get_logger().info("Trajectory accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self.get_logger().info("Trajectory execution completed.")


def main(args=None):
    rclpy.init(args=args)
    node = VelocityToTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
