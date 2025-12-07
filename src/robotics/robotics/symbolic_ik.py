#!/usr/bin/env python3
"""
Symbolic IK solver matching MATLAB approach exactly
Derives R33 symbolically from DH parameters
"""

import numpy as np
import sympy as sp
from sympy import symbols, cos, sin, simplify, solve, pi, Matrix, Eq


class FK:
    def __init__(self):
        self.l1 = 41.05
        self.l2 = 139.93
        self.l3 = 132.9
        self.l4 = 52.3
        self.l5 = 37.76
    
    def fwd_kinematics(self, q):
        l1, l2, l3= self.l1, self.l2, self.l3
        theta = [-np.pi/2+q[0], -np.pi/2+q[1], q[2]]
        d = [-l1, 0, 0]
        a = [0, l2, l3, 0, 0]
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

# Test
fk = FK()

class SymbolicIKMATLAB:
    def __init__(self):
        # Link lengths (mm)
        self.l1 = 41.05
        self.l2 = 139.93
        self.l3 = 132.9
        self.l4 = 52.3
        self.l5 = 37.76

        # Joint limits (radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # q1
            [-np.pi/2, np.pi/2],  # q2
            [-np.pi/2, np.pi/2],  # q3
            [-np.pi/2, np.pi/2],  # q4
            [-np.pi, np.pi]       # q5
        ])
        
        print("Symbolic IK Solver (MATLAB-style) initialized")
        print(f"Link lengths: l1={self.l1}, l2={self.l2}, l3={self.l3}")
        
        # Derive R33 symbolically once
        # print("\nDeriving symbolic R33 expression from DH parameters...")
        # self.R33_symbolic = self.derive_R33_symbolic()
        # print("✓ R33 expression derived")
    
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


    def last_joints_inv_kinematics(self, q1_val, q2_val, q3_val, direction=-1.0):
        """
        Compute q4 (last joint) and q5=0 for the manipulator using symbolic R33.
        This mirrors MATLAB-style symbolic computation.
        
        Args:
            q1_val, q2_val, q3_val: joint angles in radians
            direction: desired R33 value (-1 for down, +1 for up)
            
        Returns:
            sol_q4: list of possible solutions for q4 (radians)
            q5: fixed to 0
        """
        # q5 is always zero
        q5 = 0.0
        
        # Define symbolic variable q4
        q4 = symbols('q4', real=True)

        # Define the symbolic R33 expression (as in MATLAB)
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

        print(f"Substituting q1={q1_val:.4f}, q2={q2_val:.4f}, q3={q3_val:.4f}")
        
        # Simplify the expression
        R33_simple = simplify(R33)
        print("Simplified R33 expression:")
        sp.pprint(R33_simple)
        
        # Solve symbolically: R33 = direction
        solutions = solve(Eq(R33_simple, direction), q4, dict=True)
        
        # Extract numeric q4 solutions
        sol_q4 = [sol[q4].evalf() for sol in solutions]
        
        print("q4 solutions (radians):", sol_q4)
        
        return sol_q4, q5

    

    def jacobian_3dof(self, q):
        """Compute 3x3 Jacobian for position w.r.t. q1, q2, q3"""
        epsilon = 1e-6
        J = np.zeros((3, 3))
        
        # Current position
        q_full = np.array([q[0], q[1], q[2], 0.0, 0.0])
        pos_current = fk.fwd_kinematics(q_full.tolist())[:3, 3]
        
        for i in range(3):
            q_perturbed = q.copy()
            q_perturbed[i] += epsilon
            q_full_perturbed = np.array([q_perturbed[0], q_perturbed[1], q_perturbed[2], 0.0, 0.0])
            pos_perturbed = fk.fwd_kinematics(q_full_perturbed.tolist())[:3, 3]
            J[:, i] = (pos_perturbed - pos_current) / epsilon
        
        return J

    def solve_ik_3dof_position(self, wrist_target, q_init):
        """
        Solve 3-DOF wrist IK using Newton-Raphson iteration with joint limits.
        
        Args:
            wrist_target: 3-element array [x, y, z] target for wrist center
            q_init: initial guess for [q1, q2, q3] in radians
        
        Returns:
            q: solution array [q1, q2, q3] (radians) if converged
            success: True if converged, False otherwise
        """
        max_iters = 1000
        tol = 0.5  # mm
        max_step = 0.1  # rad per iteration
        damp_lambda = 1e-2

        q = np.array(q_init, dtype=float)

        for it in range(max_iters):
            # Forward kinematics
            q_full = [q[0], q[1], q[2]]
            pos_current = fk.fwd_kinematics(q_full)[:3, 3]

            # Position error
            error_vec = pos_current - np.array(wrist_target)
            err_norm = np.linalg.norm(error_vec)
            if err_norm < tol:
                return q[0], q[1], q[2], True

            # Jacobian
            J = self.jacobian_3dof(q)

            # Damped pseudoinverse
            JJt = J @ J.T
            inv_term = np.linalg.inv(JJt + (damp_lambda**2) * np.eye(3))
            J_damped_pinv = J.T @ inv_term

            # Newton-Raphson update
            dq = -J_damped_pinv @ error_vec

            # Limit step size
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_step:
                dq = dq * (max_step / dq_norm)

            # Update q and apply joint limits
            q += dq
            q = self.apply_joint_limits(q)

        # Failed to converge
        return q[0], q[1], q[2], False

    def fk_numerical(self, q):
        """Forward kinematics"""
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
    
    def solve_ik(self, wrist_target, q5=0.0, dir_target=-1.0):
        """
        Complete IK solution (MATLAB approach)
        
        Args:
            ee_target: [x, y, z] in mm
            q5: rotation angle (radians)
            dir_target: desired R33 (-1 for down, +1 for up)
        """
        print(f"\n{'='*70}")
        print(f"COMPLETE IK SOLUTION (MATLAB APPROACH)")
        print(f"{'='*70}")
        print(f"Target EE: [{wrist_target[0]:.1f}, {wrist_target[1]:.1f}, {wrist_target[2]:.1f}] mm")
        print(f"Target dir (R33): {dir_target:.1f}")
        print(f"q5: {np.rad2deg(q5):.1f}°")
        
        # # Step 1: Compute wrist center
        # tool_length = self.l4 + self.l5
        # tool_dir = np.array([0, 0, -1]) if dir_target < 0 else np.array([0, 0, 1])
        wrist = wrist_target
        
        # Step 2: Solve 3-DOF
        q1, q2, q3, success = self.solve_ik_3dof_position(wrist, [0,0,0])

        if not success:
             q1, q2, q3, success = self.solve_ik_3dof_position(wrist, [np.pi/4, -np.pi/4, np.pi/4])

        if not success:
            print("⚠ 3-DOF IK did not converge")
        
        if q1 is None:
            print(f"\n{'='*70}")
            print("✗ IK FAILED: 3-DOF solution not found")
            print(f"{'='*70}")
            return None
        
        # Step 3: Solve q4 symbolically
        q4_list, q5 = self.last_joints_inv_kinematics(q1, q2, q3, dir_target)
        if not q4_list:
            print("⚠ No solution for q4!")
            return None
        q4 = float(q4_list[0])  # pick the first solution and convert to float

        q_solution = np.array([q1, q2, q3, q4, q5])

        
        # Verify
        print(f"\n{'='*70}")
        print("VERIFICATION")
        print(f"{'='*70}")
        T = self.fk_numerical(q_solution)
        wrist_achieved = T[:3, 3]
        R33_achieved = T[2, 2]
        
        ee_error = np.linalg.norm(wrist_achieved - wrist_target)
        R33_error = abs(R33_achieved - dir_target)
        
        print(f"Solution: q = {np.round(np.rad2deg(q_solution), 2)}°")
        print(f"EE achieved: [{wrist_achieved[0]:.1f}, {wrist_achieved[1]:.1f}, {wrist_achieved[2]:.1f}] mm")
        print(f"EE error: {ee_error:.3f} mm")
        print(f"R33 achieved: {R33_achieved:.6f} (target: {dir_target:.1f})")
        print(f"R33 error: {R33_error:.6f}")
        
        if ee_error < 5.0 and R33_error < 0.1:
            print("✓ IK CONVERGED SUCCESSFULLY!")
        elif ee_error < 10.0:
            print("⚠ IK PARTIAL SUCCESS")
        else:
            print("✗ IK FAILED")
        
        print(f"{'='*70}\n")
        return q_solution


def test():
    """Test cases"""
    ik = SymbolicIKMATLAB()
    
    # Test 1: Forward position
    print("\n" + "#"*70)
    print("TEST 1: Forward position")
    print("#"*70)
    q1 = ik.solve_ik([0, 0, -313.88], q5=0, dir_target=1.0)
    
    # Test 2: Side position
    print("\n" + "#"*70)
    print("TEST 2: Side position")
    print("#"*70)
    q2 = ik.solve_ik([0, 250, -110], q5=0, dir_target=1.0)
    
    # Test 3: Home position (singularity)
    print("\n" + "#"*70)
    print("TEST 3: Home position (singularity)")
    print("#"*70)
    q3 = ik.solve_ik([0, 0, -250], q5=0, dir_target=1.0)


if __name__ == "__main__":
    test()