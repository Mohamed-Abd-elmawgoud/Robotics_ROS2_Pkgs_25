#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
import time
from robotics.fwd_kinematics import ForwardKinematicsNode


class VelocityValidator(Node):
    def __init__(self):
        super().__init__('velocity_validator')
        
        # Your kinematics
        self.fk_node = ForwardKinematicsNode()
        
        # Gazebo data
        self.joint_positions = None
        self.joint_velocities = None
        self.prev_ee_pos = None
        self.prev_time = None
        self.gazebo_cart_vel = None
        
        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Timer for comparison (0.1 seconds = 10 Hz)
        self.timer = self.create_timer(0.1, self.compare_velocities)
        
        self.get_logger().info('Velocity Validator Node Started')
        
    def joint_callback(self, msg):
        """Store joint data and compute end-effector velocity"""
        # Store joint positions and velocities
        self.joint_positions = np.array(msg.position[:5])
        self.joint_velocities = np.array(msg.velocity[:5])
        
        # Compute end-effector position using forward kinematics
        T = self.fk_node.fwd_kinematics(self.joint_positions.tolist())
        current_ee_pos = T[:3, 3]  # Extract position (x, y, z) in mm
        
        current_time = time.time()
        
        # Compute velocity using numerical differentiation
        if self.prev_ee_pos is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                self.gazebo_cart_vel = (current_ee_pos - self.prev_ee_pos) / dt
        
        # Store for next iteration
        self.prev_ee_pos = current_ee_pos
        self.prev_time = current_time
    
    def compare_velocities(self):
        """Compare theoretical (Jacobian) vs measured (numerical derivative) velocities"""
        # Wait until we have all necessary data
        if (self.joint_positions is None or 
            self.joint_velocities is None or 
            self.gazebo_cart_vel is None):
            return
        
        # Compute theoretical Cartesian velocity using Jacobian
        J = self.jacobian_matrix(self.joint_positions)
        computed_cart_vel = J @ self.joint_velocities
        
        # Compute error
        error = np.linalg.norm(computed_cart_vel - self.gazebo_cart_vel)
        gazebo_vel_norm = np.linalg.norm(self.gazebo_cart_vel)
        error_percent = (error / (gazebo_vel_norm + 1e-6)) * 100
        
        # Print comparison
        self.get_logger().info("="*60)
        self.get_logger().info(f"Joint positions (rad): {np.round(self.joint_positions, 3)}")
        self.get_logger().info(f"Joint velocities (rad/s): {np.round(self.joint_velocities, 4)}")
        self.get_logger().info(f"Computed velocity (Jacobian): {np.round(computed_cart_vel, 2)} mm/s")
        self.get_logger().info(f"Measured velocity (Numerical): {np.round(self.gazebo_cart_vel, 2)} mm/s")
        self.get_logger().info(f"Absolute error: {error:.4f} mm/s ({error_percent:.2f}%)")
        self.get_logger().info("="*60)
    
    def jacobian_matrix(self, q):
        """Compute Jacobian using numerical differentiation"""
        epsilon = 1e-6
        J = np.zeros((3, 5))
        pos_current = self.fk_node.fwd_kinematics(q.tolist())[:3, 3]
        
        for i in range(5):
            q_pert = q.copy()
            q_pert[i] += epsilon
            pos_pert = self.fk_node.fwd_kinematics(q_pert.tolist())[:3, 3]
            J[:, i] = (pos_pert - pos_current) / epsilon
        
        return J


def main(args=None):
    rclpy.init(args=args)
    
    validator = VelocityValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()