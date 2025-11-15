#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import numpy as np

# Try different import patterns depending on package structure
try:
    from robotics.fwd_kinematics import ForwardKinematicsNode
except ImportError:
    try:
        from fwd_kinematics import ForwardKinematicsNode
    except ImportError:
        # If both fail, we'll define a minimal FK class here
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
                        [ct, -st * ca,  st * sa, a[i] * ct],
                        [st,  ct * ca, -ct * sa, a[i] * st],
                        [0,   sa,       ca,      d[i]],
                        [0,   0,        0,       1]
                    ])
                    
                    T = T @ T_i
                
                return T


class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('inverse_kinematics_node')

        # Create an instance of ForwardKinematicsNode to use its fwd_kinematics method
        # We don't spin it, just use it as a library
        self.fk_node = ForwardKinematicsNode()
        
        # Newton-Raphson parameters
        self.error_tolerance = 0.001
        self.max_iterations = 1000

        # Joint limits (radians) - adjust these based on your robot's actual limits
        # Format: [min, max] for each joint
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # q1: ±180°
            [-np.pi/2, np.pi/2],  # q2: ±90° (prevent extreme bending)
            [-np.pi/2, np.pi/2],  # q3: ±90° (prevent extreme bending)
            [-np.pi/2, np.pi/2],  # q4: ±90°
            [-np.pi, np.pi]       # q5: ±180°
        ])

        # Current joint angles (initial guess) - set to middle of joint ranges
        self.q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All within limits

        # Subscribers and publishers
        self.position_sub = self.create_subscription(
            Point,
            'desired_position',
            self.position_callback,
            10
        )

        self.joint_angles_pub = self.create_publisher(
            Float64MultiArray,
            'joint_angles',
            10
        )

        self.get_logger().info("Inverse Kinematics Node started (5-DOF, position only).")

    def normalize_angles(self, q):
        """
        Normalize angles to [-pi, pi] range
        
        Input:
            q - numpy array of joint angles
        Output:
            q_normalized - angles wrapped to [-pi, pi]
        """
        q_normalized = np.copy(q)
        for i in range(len(q)):
            # Wrap to [-pi, pi]
            q_normalized[i] = np.arctan2(np.sin(q[i]), np.cos(q[i]))
        return q_normalized

    def apply_joint_limits(self, q):
        """
        Apply joint limits by first normalizing then clamping angles to valid range
        
        Input:
            q - numpy array of joint angles
        Output:
            q_limited - angles normalized and clamped to joint limits
        """
        q_limited = self.normalize_angles(q)  # First normalize to [-pi, pi]
        for i in range(len(q_limited)):
            q_limited[i] = np.clip(q_limited[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        return q_limited

    def check_joint_limits(self, q):
        """
        Check if joint angles are within limits
        
        Input:
            q - numpy array of joint angles
        Output:
            valid - boolean indicating if all joints are within limits
        """
        for i in range(len(q)):
            if q[i] < self.joint_limits[i, 0] or q[i] > self.joint_limits[i, 1]:
                return False
        return True

    # ------------------------------------------------------------
    # Jacobian matrix computation (numerical)
    # ------------------------------------------------------------
    def jacobian_matrix(self, q):
        """
        Computes the Jacobian matrix numerically using finite differences
        
        Input:
            q - numpy array of 5 joint angles
        Output:
            J - 3x5 Jacobian matrix (position only)
        """
        epsilon = 1e-6
        J = np.zeros((3, 5))
        
        # Get current position using imported FK (returns mm)
        T_current = self.fk_node.fwd_kinematics(q.tolist())
        pos_current = T_current[:3, 3]  # This is in mm
        
        # Compute partial derivatives numerically
        for i in range(5):
            q_perturbed = q.copy()
            q_perturbed[i] += epsilon
            T_perturbed = self.fk_node.fwd_kinematics(q_perturbed.tolist())
            pos_perturbed = T_perturbed[:3, 3]  # This is in mm
            
            # Partial derivative (mm per radian)
            J[:, i] = (pos_perturbed - pos_current) / epsilon
        
        return J

    # ------------------------------------------------------------
    # Inverse Jacobian (pseudo-inverse with singularity handling)
    # ------------------------------------------------------------
    def inverse_jacobian_matrix(self, q):
        """
        Computes the pseudo-inverse of the Jacobian matrix
        
        Input:
            q - numpy array of 5 joint angles
        Output:
            J_inv - 5x3 pseudo-inverse of Jacobian
        """
        J = self.jacobian_matrix(q)
        
        # Check for singularity
        JJT = J @ J.T
        det_JJT = np.linalg.det(JJT)
        
        if abs(det_JJT) < 1e-6:
            # Near singularity - perturb joints slightly
            self.get_logger().warn("Near singularity detected, perturbing joint angles")
            q[1] += 0.1
            q[2] += 0.1
            J = self.jacobian_matrix(q)
            JJT = J @ J.T
        
        # Compute pseudo-inverse: J^+ = J^T * (J * J^T)^-1
        try:
            J_inv = J.T @ np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            self.get_logger().error("Failed to compute Jacobian inverse")
            J_inv = np.linalg.pinv(J)  # Fallback to standard pseudo-inverse
        
        return J_inv

    # ------------------------------------------------------------
    # Inverse kinematics using Newton-Raphson
    # ------------------------------------------------------------
    def inv_kinematics(self, q0, X_desired):
        """
        Solves inverse kinematics using Newton-Raphson method
        
        Input:
            q0 - initial guess for joint angles (5x1)
            X_desired - desired end-effector position in mm (3x1)
        Output:
            q_final - final joint angles (5x1)
            converged - boolean indicating if solution converged
        """
        q_current = q0.copy()
        
        for iteration in range(self.max_iterations):
            # Compute current position using imported FK
            T_current = self.fk_node.fwd_kinematics(q_current.tolist())
            pos_current = T_current[:3, 3]
            
            # Error function: F(q) = current_position - desired_position
            F_current = pos_current - X_desired
            
            # Check convergence
            error_norm = np.linalg.norm(F_current)
            if error_norm < self.error_tolerance:
                self.get_logger().info(
                    f"Converged in {iteration} iterations with error = {error_norm:.6f} mm"
                )
                return q_current, True
            
            # Compute inverse Jacobian
            J_inv = self.inverse_jacobian_matrix(q_current)
            
            # Newton-Raphson update: q_{n+1} = q_n - J^{-1} * F(q_n)
            delta_q = -J_inv @ F_current
            
            # Update joint angles
            q_current = q_current + delta_q
            
            # Log progress every 100 iterations
            if iteration % 100 == 0:
                self.get_logger().info(f"Iteration {iteration}: error = {error_norm:.6f} mm")
        
        # Maximum iterations reached
        self.get_logger().warn(
            f"Maximum iterations ({self.max_iterations}) reached. Solution may not have converged."
        )
        return q_current, False

    # ------------------------------------------------------------
    # Callback for desired position
    # ------------------------------------------------------------
    def position_callback(self, msg):
        # Desired position in meters, convert to mm
        X_desired = np.array([msg.x * 1000.0, msg.y * 1000.0, msg.z * 1000.0])
        
        self.get_logger().info(f"Received desired position: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}] m")
        self.get_logger().info(f"Desired position in mm: [{X_desired[0]:.2f}, {X_desired[1]:.2f}, {X_desired[2]:.2f}]")
        
        # Check if position is reachable (rough workspace check)
        # Robot height when standing: 403 mm (given)
        # This is approximately l1 + l2 + l3 + l4 + l5 when fully extended
        max_reach = 403.0  # mm - robot's maximum reach when fully extended
        position_radius = np.linalg.norm(X_desired)
        
        if position_radius > max_reach:
            self.get_logger().error(
                f"Position out of reach! Distance: {position_radius:.2f} mm, Max reach: {max_reach:.2f} mm"
            )
            return
        
        # If current guess is all zeros, provide a better initial guess
        if np.allclose(self.q_current, 0):
            # Use safe initial guess - middle of joint ranges or slight bend
            self.q_current = np.array([0.0, 0.2, 0.2, 0.2, 0.0])
            self.get_logger().info("Using safe initial guess within joint limits")
        
        # Ensure current guess is within limits
        self.q_current = self.apply_joint_limits(self.q_current)
        
        # Normalize the current guess before solving
        self.q_current = self.normalize_angles(self.q_current)
        
        # Solve inverse kinematics
        q_solution, converged = self.inv_kinematics(self.q_current, X_desired)
        
        if not converged:
            self.get_logger().error("Failed to find IK solution within iteration limit - NOT publishing")
            return
        
        # Verify the solution by computing forward kinematics
        T_verify = self.fk_node.fwd_kinematics(q_solution.tolist())
        pos_verify = T_verify[:3, 3]
        verification_error = np.linalg.norm(pos_verify - X_desired)
        
        self.get_logger().info(f"Verification error: {verification_error:.6f} mm")
        self.get_logger().info(f"Achieved position in mm: [{pos_verify[0]:.2f}, {pos_verify[1]:.2f}, {pos_verify[2]:.2f}]")
        
        # Update current joint angles for next iteration
        self.q_current = q_solution
        
        # Publish joint angles
        joint_msg = Float64MultiArray()
        joint_msg.data = q_solution.tolist()
        self.joint_angles_pub.publish(joint_msg)
        
        self.get_logger().info(
            f"Solution - Joint angles (rad): {np.array2string(q_solution, precision=4)}"
        )
        self.get_logger().info(
            f"Solution - Joint angles (deg): {np.array2string(np.rad2deg(q_solution), precision=2)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()