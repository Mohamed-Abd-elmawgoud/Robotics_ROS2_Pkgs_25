#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
import time

class InverseVelocityValidator(Node):
    def __init__(self):
        super().__init__('inverse_velocity_validator')
        
        # Your kinematics
        self.fk_node = ForwardKinematicsNode()
        
        # State variables
        self.joint_positions = None
        self.gazebo_joint_velocities = None
        self.prev_ee_pos = None
        self.prev_time = None
        self.measured_cart_vel = None
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Timer for comparison
        self.timer = self.create_timer(0.1, self.compare_velocities)
        
        self.get_logger().info('Inverse Velocity Validator Node Started')
        self.get_logger().info('This will compare:')
        self.get_logger().info('  1. Joint velocities from Gazebo (q̇_gazebo)')
        self.get_logger().info('  2. Joint velocities from YOUR inverse kinematics (q̇_computed)')
        self.get_logger().info('     computed from measured Cartesian velocity\n')
    
    def joint_callback(self, msg):
        """Store joint data and compute measured Cartesian velocity"""
        # Store joint positions and velocities from Gazebo
        self.joint_positions = np.array(msg.position[:5])
        self.gazebo_joint_velocities = np.array(msg.velocity[:5])
        
        # Compute end-effector position using forward kinematics
        T = self.fk_node.fwd_kinematics(self.joint_positions.tolist())
        current_ee_pos = T[:3, 3]  # Extract position (x, y, z) in mm
        
        current_time = time.time()
        
        # Compute Cartesian velocity using numerical differentiation
        if self.prev_ee_pos is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                self.measured_cart_vel = (current_ee_pos - self.prev_ee_pos) / dt
        
        # Store for next iteration
        self.prev_ee_pos = current_ee_pos
        self.prev_time = current_time
    
    def compare_velocities(self):
        """Compare Gazebo joint velocities vs computed joint velocities"""
        # Wait until we have all necessary data
        if (self.joint_positions is None or 
            self.gazebo_joint_velocities is None or 
            self.measured_cart_vel is None):
            return
        
        # Method 1: Joint velocities directly from Gazebo
        q_dot_gazebo = self.gazebo_joint_velocities
        
        # Method 2: Compute joint velocities using YOUR inverse velocity kinematics
        # Given the measured Cartesian velocity, what joint velocities should produce it?
        q_dot_computed = self.inv_velocity_kinematics(
            self.joint_positions, 
            self.measured_cart_vel
        )
        
        # IGNORE LAST JOINT (Joint 5) - it doesn't affect position
        # Only compare first 4 joints
        q_dot_gazebo_relevant = q_dot_gazebo[:4]
        q_dot_computed_relevant = q_dot_computed[:4]
        
        # Compute error only for relevant joints
        error = np.linalg.norm(q_dot_gazebo_relevant - q_dot_computed_relevant)
        gazebo_norm = np.linalg.norm(q_dot_gazebo_relevant)
        error_percent = (error / (gazebo_norm + 1e-6)) * 100
        
        # Component-wise errors (all 5 joints for display)
        component_errors = np.abs(q_dot_gazebo - q_dot_computed)
        
        # Print comparison
        self.get_logger().info("="*70)
        self.get_logger().info(f"Joint positions (rad):           {np.round(self.joint_positions, 3)}")
        self.get_logger().info(f"Measured Cartesian vel (mm/s):   {np.round(self.measured_cart_vel, 2)}")
        self.get_logger().info("-"*70)
        self.get_logger().info(f"Gazebo joint velocities (rad/s): {np.round(q_dot_gazebo, 4)}")
        self.get_logger().info(f"Computed joint vel (YOUR IK):    {np.round(q_dot_computed, 4)}")
        self.get_logger().info("-"*70)
        self.get_logger().info(f"Component-wise errors (rad/s):   {np.round(component_errors, 4)}")
        self.get_logger().info(f"Total error (joints 1-4 only): {error:.6f} rad/s ({error_percent:.2f}%)")
        self.get_logger().info(f"Note: Joint 5 ignored (doesn't affect position)")
        self.get_logger().info("="*70 + '\n')
    
    # ----------------------------------------
    # Inverse velocity kinematics (YOUR CODE)
    # ----------------------------------------
    def inv_velocity_kinematics(self, q, X_dot):
        """Compute joint velocities from Cartesian velocity using YOUR inverse kinematics"""
        J_inv = self.inverse_jacobian_matrix(q)
        return J_inv @ X_dot
    
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
    
    def inverse_jacobian_matrix(self, q):
        """Compute pseudo-inverse of Jacobian"""
        J = self.jacobian_matrix(q)
        JJT = J @ J.T
        
        # Check for singularity
        if abs(np.linalg.det(JJT)) < 1e-6:
            self.get_logger().warn('Near singularity detected! Perturbing configuration...')
            q = q.copy()
            q[1] += 0.1
            q[2] += 0.1
            J = self.jacobian_matrix(q)
            JJT = J @ J.T
        
        try:
            # J^T * (J * J^T)^-1
            return J.T @ np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            self.get_logger().warn('Using pseudo-inverse fallback')
            return np.linalg.pinv(J)


# FK class
class ForwardKinematicsNode:
    def __init__(self):
        self.l1 = 41.05
        self.l2 = 139.93
        self.l3 = 132.9
        self.l4 = 52.3
        self.l5 = 37.76

    def fwd_kinematics(self, q):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        theta = [
            -np.pi/2 + q[0],
            -np.pi/2 + q[1],
            q[2],
            -np.pi/2 + q[3],
            np.pi/2 + q[4]
        ]
        d = [-l1, 0, 0, 0, -l4-l5]
        a = [0, l2, l3, 0, 0]
        alpha = [np.pi/2, np.pi, np.pi, np.pi/2, 0]
        T = np.eye(4)
        for i in range(5):
            ct = np.cos(theta[i])
            st = np.sin(theta[i])
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])
            T_i = np.array([
                [ct, -st*ca, st*sa, a[i]*ct],
                [st, ct*ca, -ct*sa, a[i]*st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
        return T


def main(args=None):
    rclpy.init(args=args)
    
    validator = InverseVelocityValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()