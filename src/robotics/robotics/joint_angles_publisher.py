#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math

class JointAnglesPublisher(Node):
    def __init__(self):
        super().__init__('joint_angles_publisher')
        self.pub = self.create_publisher(Float64MultiArray, '/joint_angles', 10)
        self.get_logger().info("Joint Angles Publisher started.")

        # Prompt user for angles in degrees
        angles_deg = input("Enter 6 joint angles in degrees, separated by spaces: ")
        angles_deg_list = [float(a) for a in angles_deg.strip().split()]

        if len(angles_deg_list) != 6:
            self.get_logger().error("You must enter exactly 6 joint angles!")
            rclpy.shutdown()
            return

        # Convert to radians
        angles_rad = [math.radians(a) for a in angles_deg_list]

        # Publish message
        msg = Float64MultiArray()
        msg.data = angles_rad
        self.pub.publish(msg)
        self.get_logger().info(f"Published joint angles (radians): {angles_rad}")


def main(args=None):
    rclpy.init(args=args)
    node = JointAnglesPublisher()
    # Give a short time to ensure the message is sent before shutdown
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
