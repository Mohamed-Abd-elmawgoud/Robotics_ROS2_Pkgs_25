#!/usr/bin/env python3
import rclpy
from std_msgs.msg import Float64
import time

def main():
    rclpy.init()
    node = rclpy.create_node('arm_mover')
    
    pub_j2 = node.create_publisher(Float64, '/joint_cmd/prismatic_j2', 10)
    pub_j3 = node.create_publisher(Float64, '/joint_cmd/prismatic_j3', 10)
    pub_j4 = node.create_publisher(Float64, '/joint_cmd/prismatic_j4', 10)
    pub_j5 = node.create_publisher(Float64, '/joint_cmd/prismatic_j5', 10)
    
    time.sleep(1)  # Wait for publishers to connect
    
    # Example: Move joints in sequence
    joints = [pub_j2, pub_j3, pub_j4, pub_j5]
    
    for i in range(5):
        for pub in joints:
            msg = Float64()
            msg.data = 0.5
            pub.publish(msg)
            time.sleep(0.2)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()