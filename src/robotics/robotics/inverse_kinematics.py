#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import numpy as np
import sympy as sp
from sympy import symbols, cos, sin, simplify, solve, pi, Eq

# Try importing FK node; fallback if not found
try:
    from robotics.fwd_kinematics import ForwardKinematicsNode
except ImportError:
    try:
        from fwd_kinematics import ForwardKinematicsNode
    except ImportError:
        class ForwardKinematicsNode:
            def __init__(self):
                self.l1 = 41.05
                self.l2 = 139.93
                self.l3 = 132.9
                self.l4 = 52.3
                self.l5 = 37.76
            
            def fwd_kinematics(self, q):
                l1, l2, l3 = self.l1, self.l2, self.l3
                theta = [-np.pi/2+q[0], -np.pi/2+q[1], q[2]]
                d = [-l1, 0, 0]
                a = [0, l2, l3]
                alpha = [np.pi/2, np.pi, np.pi]
                T = np.eye(4)
                for i in range(3):
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


class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('inverse_kinematics_node')

        # FK node instance
        self.fk_node = ForwardKinematicsNode()
        
        # Link lengths
        self.l1 = self.fk_node.l1
        self.l2 = self.fk_node.l2
        self.l3 = self.fk_node.l3
        self.l4 = self.fk_node.l4
        self.l5 = self.fk_node.l5

        # IK parameters
        self.error_tolerance = 0.5  # mm
        self.max_iterations = 10000

        # Joint limits (radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # q1
            [-np.pi/2, np.pi/2],  # q2
            [-np.pi/2, np.pi/2],  # q3
            [-np.pi/2, np.pi/2],  # q4
            [-np.pi, np.pi]       # q5
        ])

        # Current joint angles
        self.q_current = np.array([0.0, 0.0, 0.0])

        self.get_logger().info("Inverse Kinematics Node started (with SYMBOLIC q4 solver)")
        self.get_logger().info("Mode: Tool ALWAYS pointing DOWN + q5 rotation")

    # ------------------------------------------------------------
    # Normalize angles to [-pi, pi]
    # ------------------------------------------------------------
    def normalize_angles(self, q):
        return np.arctan2(np.sin(q), np.cos(q))

    # ------------------------------------------------------------
    # Apply joint limits
    # ------------------------------------------------------------
    def apply_joint_limits(self, q):
        q = self.normalize_angles(q)
        for i in range(len(q)):
            q[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        return q
    

    def inverse_jacobian_matrix(self, q):
        J = self.jacobian_3dof(q)
        try:
            J_inv = J.T @ np.linalg.inv(J @ J.T)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular Jacobian, using np.linalg.pinv fallback")
            J_inv = np.linalg.pinv(J)
        return J_inv

    # ------------------------------------------------------------
    # Compute numerical Jacobian for position (3-DOF: q1, q2, q3)
    # ------------------------------------------------------------
    def jacobian_3dof(self, q):
        """Compute 3x3 Jacobian for position w.r.t. q1, q2, q3"""
        epsilon = 1e-6
        J = np.zeros((3, 3))
        
        # Current position
        q_full = np.array([q[0], q[1], q[2]])
        pos_current = self.fk_node.fwd_kinematics(q_full.tolist())[:3, 3]
        
        for i in range(3):
            q_perturbed = q.copy()
            q_perturbed[i] += epsilon
            q_full_perturbed = np.array([q_perturbed[0], q_perturbed[1], q_perturbed[2]])
            pos_perturbed = self.fk_node.fwd_kinematics(q_full_perturbed.tolist())[:3, 3]
            J[:, i] = (pos_perturbed - pos_current) / epsilon
        
        return J
    
    # ------------------------------------------------------------
    # Get R33 element from rotation matrix
    # ------------------------------------------------------------
    def get_R33(self, q):
        """Get R33 element (3,3) of rotation matrix"""
        T = self.fk_node.fwd_kinematics(q.tolist())
        return T[2, 2]
    
     # ------------------------------------------------------------
    # NUMERIC q4 solver
    # ------------------------------------------------------------
    def solve_q4_numeric(self, q1_val, q2_val, q3_val, direction=-1.0):
        """
        Compute q4 such that R33 = direction (+1 or -1) using numeric formula.
        """
        try:
            c2 = np.cos(q2_val - np.pi/2)
            s2 = np.sin(q2_val - np.pi/2)
            c3 = np.cos(q3_val)
            s3 = np.sin(q3_val)

            # Coefficients for R33 = A*cos(q4) + B*sin(q4)
            A = c2*c3 - s2*s3
            B = -c2*s3 - s2*c3

            R = np.hypot(A, B)
            if R < 1e-6:
                q4_val = 0.0
                success = False
            else:
                phi = np.arctan2(B, A)
                ratio = direction / R

                # Clamp to valid domain of arcsin
                ratio = np.clip(ratio, -1.0, 1.0)

                angle = np.arcsin(ratio)

                q4_1 = angle - phi
                q4_2 = np.pi - angle - phi

                q4_1 = (q4_1 + np.pi) % (2*np.pi) - np.pi
                q4_2 = (q4_2 + np.pi) % (2*np.pi) - np.pi
                q4_val = q4_1 if abs(q4_1) < abs(q4_2) else q4_2
                success = True

            # Clip to joint limits
            q4_val = self.normalize_angles(q4_val)
            q4_val = np.clip(q4_val, self.joint_limits[3, 0], self.joint_limits[3, 1])

            # Optional verification
            q_verify = np.array([q1_val, q2_val, q3_val, q4_val, 0.0])
            R33_achieved = self.get_R33(q_verify)
            error = abs(R33_achieved - direction)
            if error > 0.01:
                self.get_logger().warn(
                    f"q4 solution has R33 error {error:.4f} (achieved={R33_achieved:.4f}, target={direction})"
                )

            return q4_val, success
        except Exception as e:
            self.get_logger().error(f"q4 numeric solver failed: {e}")
            return 0.0, False

   

    # ------------------------------------------------------------
    # Solve 3-DOF IK for position only (q1, q2, q3)
    # ------------------------------------------------------------
    def solve_ik_3dof_position(self, wrist_target, q_init):
        """
        Simple 3-DOF IK solver using Newton-Raphson with inverse Jacobian.
        Mirrors the style of `inv_kinematics`.
        """
        q = np.array(q_init[:3], dtype=float)

        for it in range(self.max_iterations):
            # Compute current wrist position
            q_full = np.array([q[0], q[1], q[2]])
            wrist_current = self.fk_node.fwd_kinematics(q_full.tolist())[:3, 3]

            # Position error
            error_vec = wrist_current - wrist_target
            err_norm = np.linalg.norm(error_vec)

            if err_norm < self.error_tolerance:
                self.get_logger().info(f"[3-DOF] Converged in {it} iterations, error={err_norm:.3f}mm")
                return q, True

            # Inverse Jacobian step
            J_inv = self.inverse_jacobian_matrix(q)
            q += -J_inv @ error_vec

            # Apply joint limits
            q = self.apply_joint_limits(q)

        self.get_logger().warn(f"[3-DOF] Failed to converge, final error={err_norm:.3f}mm")
        return q, False


    # ------------------------------------------------------------
    # Main IK: Position + Always pointing DOWN + q5 rotation
    # ------------------------------------------------------------
    def solve_ik_full(self, q_init, ee_target, q5_target):
        # Step 1: Compute wrist center
        tool_length = self.l4 + self.l5
        tool_dir_down = np.array([0.0, 0.0, -1.0])
        wrist_target = ee_target - tool_length * tool_dir_down

        # Step 2: Solve 3-DOF IK
        q123, success_pos = self.solve_ik_3dof_position(wrist_target, q_init)
        if not success_pos:
            self.get_logger().error("3-DOF wrist IK failed")
            return q_init, False

        # Step 3: Solve q4 numerically
        q4, success_q4 = self.solve_q4_numeric(q123[0], q123[1], q123[2], direction=-1.0)

        # Step 4: Assemble solution
        q_solution = np.array([q123[0], q123[1], q123[2], q4, q5_target])
        return q_solution, success_pos and success_q4

    # ------------------------------------------------------------
    # Simplified interface
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos, q5, q_init):
        return self.solve_ik_full(q_init, ee_pos, q5)
    

    # ------------------------------------------------------------
    # Simplified interface (for compatibility)
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos, q5, q_init):
        """Simplified interface: position + q5 only (always points down)"""
        return self.solve_ik_full(q_init, ee_pos, q5)


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