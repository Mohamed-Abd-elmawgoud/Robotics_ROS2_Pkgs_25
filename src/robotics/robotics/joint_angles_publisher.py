#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import math


class UserInputPublisher(Node):
    def __init__(self):
        super().__init__('user_input_publisher')

        # Publishers
        self.pub_angles = self.create_publisher(Float64MultiArray, '/joint_angles_out', 10)
        self.pub_velocities = self.create_publisher(Float64MultiArray, '/desired_velocities', 10)
        self.pub_position = self.create_publisher(Point, '/desired_position', 10)

        self.get_logger().info("User Input Publisher started.")

        print("\n===============================================")
        print(" Select Input Mode")
        print(" 1. Joint Angles (degrees)")
        print(" 2. Joint Velocities (rad/s) + Time From Start (s)")
        print(" 3. End-Effector Cartesian Position (meters)")
        print("===============================================\n")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            self.publish_angles()
        elif choice == "2":
            self.publish_velocities()
        elif choice == "3":
            self.publish_position()
        else:
            print("Invalid choice. Exiting.")
            rclpy.shutdown()
            return

    # -------------------------------------------------------------
    # MODE 1: Joint angles input
    # -------------------------------------------------------------
    def publish_angles(self):
        angles_deg = input("Enter 5 joint angles (degrees), separated by spaces: ")
        angles_deg_list = [float(a) for a in angles_deg.strip().split()]

        if len(angles_deg_list) != 5:
            self.get_logger().error("You must enter exactly 5 angles!")
            return

        # Convert to radians
        angles_rad = [math.radians(a) for a in angles_deg_list]

        msg = Float64MultiArray()
        msg.data = angles_rad
        self.pub_angles.publish(msg)
        self.get_logger().info(f"Published joint angles (rad): {angles_rad}")

    # -------------------------------------------------------------
    # MODE 2: Joint velocities input
    # -------------------------------------------------------------
    def publish_velocities(self):
        vel_in = input("Enter 5 joint velocities (rad/s), separated by spaces: ")
        vel_list = [float(v) for v in vel_in.strip().split()]

        if len(vel_list) != 5:
            self.get_logger().error("You must enter exactly 5 velocities!")
            return

        tfs = float(input("Enter time from start (seconds): "))

        msg = Float64MultiArray()
        # Pack velocities + time_from_start as one array
        msg.data = vel_list + [tfs]

        self.pub_velocities.publish(msg)
        self.get_logger().info(f"Published desired velocities: {vel_list}  with time_from_start={tfs}")

    # -------------------------------------------------------------
    # MODE 3: End-effector position input
    # -------------------------------------------------------------
    def publish_position(self):
        pos_in = input("Enter X Y Z (meters), separated by spaces: ")
        xyz = [float(p) for p in pos_in.strip().split()]

        if len(xyz) != 3:
            self.get_logger().error("You must enter exactly 3 numbers!")
            return

        msg = Point()
        msg.x, msg.y, msg.z = xyz

        self.pub_position.publish(msg)
        self.get_logger().info(f"Published desired EE position (m): {xyz}")


def main(args=None):
    rclpy.init(args=args)
    node = UserInputPublisher()
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
