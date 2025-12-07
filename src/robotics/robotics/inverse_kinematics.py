#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import numpy as np
from scipy.optimize import fsolve

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

        self.get_logger().info("Inverse Kinematics Node started (Decoupled 3+1+1 DOF IK)")
        self.get_logger().info("Method: 3-DOF position IK + q4 for pitch + q5 for rotation")

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
    # Solve for q4 given q1, q2, q3 to achieve target R33
    # ------------------------------------------------------------
    def solve_q4_for_R33(self, q1, q2, q3, R33_target):
        """
        Solve for q4 such that R33 = R33_target
        Similar to MATLAB solve_q4_numeric.m
        """
        def objective(q4):
            q_full = np.array([q1, q2, q3, q4, 0.0])
            R33_current = self.get_R33(q_full)
            return R33_current - R33_target
        
        # Try multiple initial guesses
        guesses = [0.0, 0.3, -0.3, 0.5, -0.5]
        
        for guess in guesses:
            try:
                q4_solution = fsolve(objective, guess, full_output=True)
                q4 = q4_solution[0][0]
                info = q4_solution[1]
                
                # Check if solution is valid
                if info['fvec'][0]**2 < 1e-6:
                    # Clip to joint limits
                    q4 = np.clip(q4, self.joint_limits[3, 0], self.joint_limits[3, 1])
                    return q4, True
            except:
                continue
        
        # If no solution found, return best guess
        return 0.0, False
    
    # ------------------------------------------------------------
    # Convert pitch angle to R33 value
    # ------------------------------------------------------------
    def pitch_to_R33(self, pitch_rad):
        """
        Convert pitch angle to desired R33 value.
        
        Pitch convention (tool orientation in X-Z plane):
        - pitch = -pi/2 (−90°) → tool pointing DOWN → R33 = -1
        - pitch = 0° → tool pointing FORWARD (+X) → R33 = 0
        - pitch = pi/2 (+90°) → tool pointing UP → R33 = 1
        
        The tool Z-axis direction is:
        tool_z = [cos(pitch), 0, sin(pitch)]
        
        So R33 (Z-component of tool Z-axis) = sin(pitch)
        """
        R33 = np.sin(pitch_rad)
        return R33
    
    # ------------------------------------------------------------
    # Get tool direction from pitch angle
    # ------------------------------------------------------------
    def get_tool_direction(self, pitch_rad):
        """
        Get tool Z-axis direction vector from pitch angle.
        
        The tool Z-axis represents the tool's pointing direction.
        R33 = -cos(pitch) is the Z-component of the tool Z-axis.
        
        For a planar robot in X-Z plane with pitch:
        - pitch = -90° → tool_z = [0, 0, -1] (pointing DOWN)
        - pitch = 0°   → tool_z = [1, 0, 0] (pointing FORWARD in +X)
        - pitch = 90°  → tool_z = [0, 0, 1] (pointing UP)
        
        The relationship is:
        tool_z = [sin(pitch + 90°), 0, -cos(pitch)]
               = [cos(pitch), 0, -cos(pitch)]  ← This doesn't work either
        
        Actually, if R33 = -cos(pitch), and we want:
        pitch = -90° → R33 = 0... NO! Let me recalculate.
        
        cos(-90°) = 0, so R33 = 0? But we want R33 = -1!
        
        Wait, let me check: for pitch = -90°:
        - We want tool pointing DOWN → tool_z = [0, 0, -1] → R33 = -1
        - R33 = -cos(pitch) = -cos(-90°) = -0 = 0 ❌ WRONG!
        
        The correct formula must be: R33 = sin(pitch)
        - pitch = -90° → R33 = sin(-90°) = -1 ✓
        - pitch = 0° → R33 = sin(0°) = 0 ✓
        - pitch = 90° → R33 = sin(90°) = 1 ✓
        
        So: tool_z = [cos(pitch), 0, sin(pitch)]
        - pitch = -90° → [0, 0, -1] ✓
        - pitch = 0° → [1, 0, 0] ✓
        - pitch = 90° → [0, 0, 1] ✓
        """
        tool_z = np.array([
            np.cos(pitch_rad),
            0,
            np.sin(pitch_rad)
        ])
        return tool_z
    
    # ------------------------------------------------------------
    # Extract pitch from R33
    # ------------------------------------------------------------
    def R33_to_pitch(self, R33):
        """
        Convert R33 value to pitch angle.
        R33 = sin(pitch) → pitch = arcsin(R33)
        """
        # Clamp R33 to [-1, 1] to avoid numerical issues with arcsin
        R33_clamped = np.clip(R33, -1.0, 1.0)
        pitch = np.arcsin(R33_clamped)
        return pitch
    
    # ------------------------------------------------------------
    # Get current pitch from joint configuration
    # ------------------------------------------------------------
    def get_current_pitch(self, q):
        """Get current pitch by extracting R33 and converting"""
        R33 = self.get_R33(q)
        return self.R33_to_pitch(R33)

    # ------------------------------------------------------------
    # Solve 3-DOF IK for position only (q1, q2, q3)
    # ------------------------------------------------------------
    def solve_ik_3dof_position(self, wrist_target, q_init):
        """
        Solve 3-DOF IK for wrist center position.
        Similar to MATLAB inv_kinematics_numeric_3DOF.m
        """
        q = q_init[:3].copy()
        
        for iteration in range(self.max_iterations):
            # Forward kinematics for first 3 joints
            q_full = np.array([q[0], q[1], q[2], 0.0, 0.0])
            T = self.fk_node.fwd_kinematics(q_full.tolist())
            wrist_current = T[:3, 3]
            
            # Position error
            error = wrist_current - wrist_target
            error_norm = np.linalg.norm(error)
            
            if error_norm < self.error_tolerance:
                return q, True
            
            # Jacobian and pseudoinverse
            J = self.jacobian_3dof(q)
            try:
                J_inv = np.linalg.pinv(J)
            except:
                return q, False
            
            # Newton step
            dq = -J_inv @ error
            
            # Limit step size
            max_step = 0.1
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_step:
                dq = dq * (max_step / dq_norm)
            
            q += dq
            q = self.apply_joint_limits(q)
        
        return q, False

    # ------------------------------------------------------------
    # Solve full 5-DOF IK (decoupled approach)
    # ------------------------------------------------------------
    def solve_ik_full(self, q_init, ee_target, pitch_target, q5_target):
        """
        Decoupled IK approach matching MATLAB:
        1. Compute wrist center from ee_target and pitch_target
        2. Solve 3-DOF IK for wrist position (q1, q2, q3)
        3. Solve for q4 to achieve desired pitch (R33 value)
        4. Set q5 directly
        """
        self.get_logger().info(f"Solving decoupled IK:")
        self.get_logger().info(f"  Target EE: {ee_target} mm")
        self.get_logger().info(f"  Target pitch: {np.rad2deg(pitch_target):.1f}°")
        
        # Step 1: Compute wrist center position
        tool_length = self.l4 + self.l5
        
        # Tool direction from desired pitch
        # R33 = -cos(pitch), so tool Z-axis is [sin(pitch), 0, -cos(pitch)]
        tool_dir = np.array([
            0,
            0,
            -np.cos(pitch_target)
        ])
        
        wrist_target = ee_target - tool_length * tool_dir
        
        self.get_logger().info(f"  Wrist target: {wrist_target} mm")
        self.get_logger().info(f"  Tool direction: {tool_dir}")
        
        # Step 2: Solve 3-DOF IK for wrist position
        q123, success_pos = self.solve_ik_3dof_position(wrist_target, q_init)
        
        if not success_pos:
            self.get_logger().warn("3-DOF position IK did not converge")
            return q_init, False
        
        self.get_logger().info(f"  3-DOF solution: q1={np.rad2deg(q123[0]):.1f}°, q2={np.rad2deg(q123[1]):.1f}°, q3={np.rad2deg(q123[2]):.1f}°")
        
        # Verify wrist position
        q_temp = np.array([q123[0], q123[1], q123[2], 0.0, 0.0])
        T_temp = self.fk_node.fwd_kinematics(q_temp.tolist())
        wrist_achieved = T_temp[:3, 3]
        wrist_error = np.linalg.norm(wrist_achieved - wrist_target)
        self.get_logger().info(f"  Wrist position error: {wrist_error:.3f} mm")
        
        # Step 3: Solve for q4 to achieve desired pitch
        R33_target = self.pitch_to_R33(pitch_target)
        self.get_logger().info(f"  Target R33: {R33_target:.4f}")
        
        q4, success_pitch = self.solve_q4_for_R33(q123[0], q123[1], q123[2], R33_target)
        
        if not success_pitch:
            self.get_logger().warn("q4 solve for pitch did not converge")
            q4 = 0.0
        
        self.get_logger().info(f"  q4 solution: {np.rad2deg(q4):.1f}°")
        
        # Step 4: Assemble full solution
        q_solution = np.array([q123[0], q123[1], q123[2], q4, q5_target])
        
        # Verify final solution
        T_final = self.fk_node.fwd_kinematics(q_solution.tolist())
        ee_achieved = T_final[:3, 3]
        ee_error = np.linalg.norm(ee_achieved - ee_target)
        
        R33_achieved = self.get_R33(q_solution)
        pitch_achieved = self.R33_to_pitch(R33_achieved)
        pitch_error = abs(pitch_target - pitch_achieved)
        
        self.get_logger().info(f"  Final EE error: {ee_error:.3f} mm")
        self.get_logger().info(f"  Achieved pitch: {np.rad2deg(pitch_achieved):.1f}° (error: {np.rad2deg(pitch_error):.2f}°)")
        self.get_logger().info(f"  Achieved R33: {R33_achieved:.4f}")
        
        # Check convergence
        success = ee_error < 5.0 and pitch_error < np.deg2rad(5.0)
        
        if success:
            self.get_logger().info("✓ Decoupled IK converged successfully!")
        else:
            self.get_logger().warn(f"⚠ Decoupled IK partial success (ee_err={ee_error:.2f}mm, pitch_err={np.rad2deg(pitch_error):.1f}°)")
        
        return q_solution, success

    # ------------------------------------------------------------
    # Solve IK for pose (simplified interface)
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos, pitch, q5, q_init):
        """
        Main IK interface
        """
        return self.solve_ik_full(q_init, ee_pos, pitch, q5)


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