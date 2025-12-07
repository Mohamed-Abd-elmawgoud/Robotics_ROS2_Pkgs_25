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
                l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
                theta = [-np.pi/2+q[0], -np.pi/2+q[1], q[2], -np.pi/2+q[3], np.pi/2+q[4]]
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
        self.max_iterations = 1000

        # Joint limits (radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # q1
            [-np.pi/2, np.pi/2],  # q2
            [-np.pi/2, np.pi/2],  # q3
            [-np.pi/2, np.pi/2],  # q4
            [-np.pi, np.pi]       # q5
        ])

        # Current joint angles
        self.q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

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

    # ------------------------------------------------------------
    # Compute numerical Jacobian for position (3-DOF: q1, q2, q3)
    # ------------------------------------------------------------
    def jacobian_3dof(self, q):
        """Compute 3x3 Jacobian for position w.r.t. q1, q2, q3"""
        epsilon = 1e-6
        J = np.zeros((3, 3))
        
        # Current position
        q_full = np.array([q[0], q[1], q[2], 0.0, 0.0])
        pos_current = self.fk_node.fwd_kinematics(q_full.tolist())[:3, 3]
        
        for i in range(3):
            q_perturbed = q.copy()
            q_perturbed[i] += epsilon
            q_full_perturbed = np.array([q_perturbed[0], q_perturbed[1], q_perturbed[2], 0.0, 0.0])
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
    # SYMBOLIC q4 solver (from MATLAB approach)
    # ------------------------------------------------------------
    def solve_q4_symbolic(self, q1_val, q2_val, q3_val, direction=1.0):
        """
        Compute q4 using symbolic R33 expression (MATLAB-style).
        
        Args:
            q1_val, q2_val, q3_val: joint angles in radians
            direction: desired R33 value (1.0 for down, -1.0 for up)
            
        Returns:
            q4_solution: float, best q4 angle (radians)
            success: True if solution found
        """
        # Define symbolic variable q4
        q4 = symbols('q4', real=True)

        # Define the symbolic R33 expression (from MATLAB DH derivation)
        R33 = ((24678615572571482867467662723121*sp.cos(q2_val - pi/2)) /
            3291009114642412084309938365114701009965471731267159726697218048 +
            (24678615572571482867467662723121*sp.sin(q3_val)*sp.sin(q2_val - pi/2)) /
            3291009114642412084309938365114701009965471731267159726697218048 +
            sp.cos(q4 - pi/2) *
            ((24678615572571482867467662723121*sp.cos(q2_val - pi/2)) /
                1645504557321206042154969182557350504982735865633579863348609024 -
                sp.sin(q3_val)*sp.sin(q2_val - pi/2) -
                sp.cos(q3_val)*(sp.cos(q2_val - pi/2) -
                24678615572571482867467662723121 /
                3291009114642412084309938365114701009965471731267159726697218048) +
                24678615572571482867467662723121 /
                3291009114642412084309938365114701009965471731267159726697218048) +
            sp.sin(q4 - pi/2) *
            (sp.cos(q3_val)*sp.sin(q2_val - pi/2) -
                sp.sin(q3_val)*(sp.cos(q2_val - pi/2) -
                24678615572571482867467662723121 /
                3291009114642412084309938365114701009965471731267159726697218048)) +
            (24678615572571482867467662723121*sp.cos(q3_val) *
                (sp.cos(q2_val - pi/2) -
                24678615572571482867467662723121 /
                3291009114642412084309938365114701009965471731267159726697218048)) /
            3291009114642412084309938365114701009965471731267159726697218048 +
            24678615572571482867467662723121 /
            6582018229284824168619876730229402019930943462534319453394436096)

        # Simplify the expression
        R33_simple = simplify(R33)
        
        # Solve symbolically: R33 = direction
        try:
            solutions = solve(Eq(R33_simple, direction), q4, dict=True)
            
            if not solutions:
                self.get_logger().warn("No symbolic solutions found for q4")
                return 0.0, False
            
            # Extract numeric q4 solutions
            q4_candidates = []
            for sol in solutions:
                try:
                    q4_val = float(sol[q4].evalf())
                    # Normalize and check limits
                    q4_val = self.normalize_angles(q4_val)
                    if self.joint_limits[3, 0] <= q4_val <= self.joint_limits[3, 1]:
                        q4_candidates.append(q4_val)
                except:
                    continue
            
            if not q4_candidates:
                self.get_logger().warn("No valid q4 solutions within joint limits")
                # Return clipped solution anyway
                q4_val = float(solutions[0][q4].evalf())
                q4_val = np.clip(self.normalize_angles(q4_val), 
                               self.joint_limits[3, 0], self.joint_limits[3, 1])
                return q4_val, False
            
            # Choose q4 closest to 0 (preferred value)
            q4_solution = min(q4_candidates, key=lambda x: abs(x))
            
            # Verify solution
            q_verify = np.array([q1_val, q2_val, q3_val, q4_solution, 0.0])
            R33_achieved = self.get_R33(q_verify)
            error = abs(R33_achieved - direction)
            
            if error > 0.1:
                self.get_logger().warn(
                    f"Symbolic q4 solution has large error: {error:.4f} "
                    f"(R33={R33_achieved:.4f}, target={direction:.4f})"
                )
            
            return q4_solution, True
            
        except Exception as e:
            self.get_logger().error(f"Symbolic solver failed: {e}")
            return 0.0, False

    # ------------------------------------------------------------
    # Solve for q4 to keep tool pointing DOWN
    # ------------------------------------------------------------
    def solve_q4_pointing_down(self, q1, q2, q3):
        """
        Solve for q4 such that the tool points straight down using symbolic solver.
        
        Target R33 = 1.0 for pointing down
        """
        R33_target = 1.0
        
        self.get_logger().info(f"  Solving symbolic q4 for R33={R33_target:.1f}...")
        
        # Use symbolic solver
        q4, success = self.solve_q4_symbolic(q1, q2, q3, direction=R33_target)
        
        if not success:
            self.get_logger().warn(f"  Symbolic solver issue, using q4={np.rad2deg(q4):.1f}°")
        else:
            self.get_logger().info(f"  ✓ Symbolic q4 solution: {np.rad2deg(q4):.1f}°")
        
        return q4

    # ------------------------------------------------------------
    # Solve 3-DOF IK for position only (q1, q2, q3)
    # ------------------------------------------------------------
    def solve_ik_3dof_position(self, wrist_target, q_init):
        """
        Robust 3-DOF IK for wrist center (q1, q2, q3).
        Uses damped pseudoinverse with multiple initial guesses.
        """

        # Solver params
        max_iters = self.max_iterations
        max_step = 0.1
        tol = self.error_tolerance
        damp_lambda = 1e-2

        # Multiple initial guesses
        base_guess = np.array(q_init[:3].copy())
        guess_list = [
            base_guess,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, -0.2, 0.2]),
            np.array([0.0, 0.3, -0.3]),
        ]

        best_q = base_guess.copy()
        best_err = np.inf

        for guess_idx, q0 in enumerate(guess_list):
            q = q0.copy()
            q = np.clip(self.normalize_angles(q), 
                       self.joint_limits[:3,0], self.joint_limits[:3,1])

            for it in range(max_iters):
                q_full = np.array([q[0], q[1], q[2]])
                T = self.fk_node.fwd_kinematics(q_full.tolist())
                wrist_current = T[:3, 3]

                error_vec = wrist_current - wrist_target
                err_norm = np.linalg.norm(error_vec)

                if err_norm < tol:
                    self.get_logger().info(
                        f"  3-DOF converged (guess {guess_idx}, iter {it}): error={err_norm:.3f}mm"
                    )
                    return q, True

                # Damped Jacobian
                J = self.jacobian_3dof(q)
                JJt = J @ J.T
                reg = (damp_lambda**2) * np.eye(3)
                try:
                    inv_term = np.linalg.inv(JJt + reg)
                    J_damped_pinv = J.T @ inv_term
                except:
                    J_damped_pinv = np.linalg.pinv(J)

                dq = -J_damped_pinv @ error_vec

                # Step limiting
                dq_norm = np.linalg.norm(dq)
                if dq_norm > max_step:
                    dq = dq * (max_step / dq_norm)

                q += dq
                q = self.apply_joint_limits(q)

            # Check final error
            final_q_full = np.array([q[0], q[1], q[2], 0.0, 0.0])
            final_pos = self.fk_node.fwd_kinematics(final_q_full.tolist())[:3, 3]
            final_err = np.linalg.norm(final_pos - wrist_target)

            if final_err < best_err:
                best_err = final_err
                best_q = q.copy()

            if best_err < tol:
                break

        if best_err < tol:
            self.get_logger().info(f"  3-DOF best error: {best_err:.3f}mm")
            return best_q, True

        self.get_logger().warn(f"  3-DOF failed: best error {best_err:.3f}mm")
        return best_q, False

    # ------------------------------------------------------------
    # Main IK: Position + Always pointing DOWN + q5 rotation
    # ------------------------------------------------------------
    def solve_ik_full(self, q_init, ee_target, q5_target):
        """
        IK matching MATLAB approach with SYMBOLIC q4 solver:
        1. Solve q1,q2,q3 for wrist center (3-link FK)
        2. Solve q4 symbolically for R33 = 1.0 (pointing down)
        
        Args:
            q_init: initial guess for joint angles
            ee_target: desired end-effector position [x, y, z] in mm
            q5_target: desired q5 angle (lamp rotation) in radians
        
        Returns:
            q_solution: solved joint angles [q1, q2, q3, q4, q5]
            success: True if converged
        """
        self.get_logger().info("="*60)
        self.get_logger().info("Solving IK (SYMBOLIC approach: wrist + symbolic q4):")
        self.get_logger().info(f"  Target EE: [{ee_target[0]:.1f}, {ee_target[1]:.1f}, {ee_target[2]:.1f}] mm")
        self.get_logger().info(f"  Target q5: {np.rad2deg(q5_target):.1f}°")
        
        # Check for singularity
        radial_dist = np.sqrt(ee_target[0]**2 + ee_target[1]**2)
        is_near_singularity = radial_dist < 10.0
        
        if is_near_singularity:
            self.get_logger().warn(f"  ⚠ Near singularity (radial dist: {radial_dist:.1f}mm)")
        
        # Step 1: Compute wrist center from EE target
        tool_length = self.l4 + self.l5
        tool_dir_down = np.array([0.0, 0.0, -1.0])
        wrist_target = ee_target - tool_length * tool_dir_down
        
        self.get_logger().info(f"  Wrist target: [{wrist_target[0]:.1f}, {wrist_target[1]:.1f}, {wrist_target[2]:.1f}] mm")
        
        # Step 2: Solve 3-DOF IK for wrist center
        q123, success_pos = self.solve_ik_3dof_position(wrist_target, q_init)
        
        if not success_pos:
            self.get_logger().error("  ✗ 3-DOF wrist IK failed!")
            return q_init, False
        
        self.get_logger().info(f"  ✓ 3-DOF solution: q=[{np.rad2deg(q123[0]):.1f}°, {np.rad2deg(q123[1]):.1f}°, {np.rad2deg(q123[2]):.1f}°]")
        
        # Step 3: Solve q4 symbolically for R33 = 1.0
        q4 = self.solve_q4_pointing_down(q123[0], q123[1], q123[2])
        
        # Step 4: Assemble full solution
        q_solution = np.array([q123[0], q123[1], q123[2], q4, q5_target])
        
        # Verify solution
        T_final = self.fk_node.fwd_kinematics(q_solution.tolist())
        ee_achieved = T_final[:3, 3]
        ee_error = np.linalg.norm(ee_achieved - ee_target)
        
        R33_achieved = self.get_R33(q_solution)
        R33_error = abs(R33_achieved - 1.0)
        
        self.get_logger().info("="*60)
        self.get_logger().info("VERIFICATION:")
        self.get_logger().info(f"  EE error: {ee_error:.3f} mm")
        self.get_logger().info(f"  R33 achieved: {R33_achieved:.4f} (target: 1.0000, error: {R33_error:.4f})")
        self.get_logger().info(f"  Final joints: {np.round(np.rad2deg(q_solution), 1)}°")
        
        # Success criteria
        if is_near_singularity:
            wrist_achieved_check = self.fk_node.fwd_kinematics([q123[0], q123[1], q123[2], 0, 0])[:3, 3]
            wrist_error = np.linalg.norm(wrist_achieved_check - wrist_target)
            success = wrist_error < 5.0
            if success:
                self.get_logger().info(f"✓ IK CONVERGED (singularity: wrist_err={wrist_error:.3f}mm)")
            else:
                self.get_logger().warn(f"⚠ IK FAILED (singularity: wrist_err={wrist_error:.3f}mm)")
        else:
            success = ee_error < 5.0 and R33_achieved > 0.90
            if success:
                self.get_logger().info("✓ IK CONVERGED SUCCESSFULLY!")
            else:
                self.get_logger().warn(f"⚠ IK PARTIAL SUCCESS")
        
        self.get_logger().info("="*60)
        return q_solution, success
    
    # ------------------------------------------------------------
    # Solve 3-DOF IK for EE position (not wrist center!)
    # ------------------------------------------------------------
    def solve_ik_3dof_ee_position(self, ee_target, q_init, q5_val):
        """
        Solve 3-DOF IK for END-EFFECTOR position (not wrist).
        With q4=0, q5=q5_val fixed.
        """
        # Reachability check
        radial = np.hypot(ee_target[0], ee_target[1])
        z_rel = ee_target[2] + self.l1
        d = np.hypot(radial, z_rel)
        
        min_r = abs(self.l2 - self.l3) 
        max_r = self.l2 + self.l3 + self.l4 + self.l5
        
        if d < min_r or d > max_r:
            self.get_logger().warn(
                f"EE target may be unreachable (distance {d:.2f} mm)"
            )
        
        # Solver params
        max_iters = self.max_iterations
        max_step = 0.1
        tol = self.error_tolerance
        damp_lambda = 1e-2

        base_guess = np.array(q_init[:3].copy())
        
        # Special case: if target is near home position, use home as guess
        if np.linalg.norm(ee_target - np.array([0, 0, -403.9])) < 10:
            guess_list = [
                np.array([0.0, 0.0, 0.0]),
                base_guess,
                np.array([0.0, -0.1, 0.1]),
                np.array([0.0, 0.1, -0.1]),
            ]
        else:
            guess_list = [
                base_guess,
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, -0.2, 0.2]),
                np.array([0.0, 0.3, -0.3]),
            ]

        best_q = base_guess.copy()
        best_err = np.inf

        for guess_idx, q0 in enumerate(guess_list):
            q = q0.copy()
            q = np.clip(self.normalize_angles(q), 
                       self.joint_limits[:3,0], self.joint_limits[:3,1])

            prev_err = np.inf
            for it in range(max_iters):
                q_full = np.array([q[0], q[1], q[2], 0.0, q5_val])
                T = self.fk_node.fwd_kinematics(q_full.tolist())
                ee_current = T[:3, 3]

                error_vec = ee_current - ee_target
                err_norm = np.linalg.norm(error_vec)

                if err_norm < tol:
                    self.get_logger().info(
                        f"  3-DOF EE converged (guess {guess_idx}, iter {it}): error={err_norm:.3f}mm"
                    )
                    return q, True
                
                if it > 0 and abs(err_norm - prev_err) < 1e-6:
                    break
                prev_err = err_norm

                # Numerical Jacobian for EE position w.r.t. q1, q2, q3
                epsilon = 1e-6
                J = np.zeros((3, 3))
                for i in range(3):
                    q_pert = q.copy()
                    q_pert[i] += epsilon
                    q_full_pert = np.array([q_pert[0], q_pert[1], q_pert[2], 0.0, q5_val])
                    ee_pert = self.fk_node.fwd_kinematics(q_full_pert.tolist())[:3, 3]
                    J[:, i] = (ee_pert - ee_current) / epsilon

                JJt = J @ J.T
                reg = (damp_lambda**2) * np.eye(3)
                try:
                    inv_term = np.linalg.inv(JJt + reg)
                    J_damped_pinv = J.T @ inv_term
                except:
                    J_damped_pinv = np.linalg.pinv(J)

                dq = -J_damped_pinv @ error_vec

                dq_norm = np.linalg.norm(dq)
                if dq_norm > max_step:
                    dq = dq * (max_step / dq_norm)

                q += dq
                q = self.apply_joint_limits(q)

            # Check final error
            final_q_full = np.array([q[0], q[1], q[2], 0.0, q5_val])
            final_pos = self.fk_node.fwd_kinematics(final_q_full.tolist())[:3, 3]
            final_err = np.linalg.norm(final_pos - ee_target)

            if final_err < best_err:
                best_err = final_err
                best_q = q.copy()

            if best_err < tol:
                break

        if best_err < 10.0:
            self.get_logger().info(f"  3-DOF EE best error: {best_err:.3f}mm (acceptable)")
            return best_q, True

        self.get_logger().warn(f"  3-DOF EE failed: best error {best_err:.3f}mm")
        return best_q, False

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