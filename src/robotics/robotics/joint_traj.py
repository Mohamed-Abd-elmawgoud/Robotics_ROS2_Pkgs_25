#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import csv
from datetime import datetime
from robotics.inverse_kinematics import InverseKinematicsNode


class PoseTrajectoryNode(Node):
    def __init__(self):
        super().__init__('pose_trajectory_node')

        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)

        # IK object (always pointing down mode)
        self.ik = InverseKinematicsNode()
        
        # Default trajectory parameters
        self.Tf = 5.0  # seconds
        self.Ts = 0.1  # seconds
        
        # Print workspace info
        self.print_workspace_info()

        # Start main loop
        self.run_loop()
    
    # ------------------------------------------------------------
    # Print workspace information
    # ------------------------------------------------------------
    def print_workspace_info(self):
        """Display approximate workspace limits"""
        l1, l2, l3, l4, l5 = self.ik.l1, self.ik.l2, self.ik.l3, self.ik.l4, self.ik.l5
        
        print("\n" + "="*70)
        print("ROBOT WORKSPACE INFORMATION")
        print("="*70)
        print(f"Link lengths (mm):")
        print(f"  l1 (base height): {l1:.2f}")
        print(f"  l2 (upper arm):   {l2:.2f}")
        print(f"  l3 (forearm):     {l3:.2f}")
        print(f"  l4 + l5 (tool):   {l4 + l5:.2f}")
        print(f"\nApproximate reach:")
        print(f"  Max horizontal: ~{l2 + l3:.1f} mm = {(l2+l3)/1000:.3f} m")
        print(f"  Max vertical (up): ~{l2 + l3 - l1:.1f} mm")
        print(f"  Max vertical (down): ~{-(l2 + l3 + l1):.1f} mm")
        print(f"  Tool length: {l4 + l5:.1f} mm")
        print(f"\n{'='*70}")
        print("IK MODE: Tool ALWAYS pointing DOWN")
        print("="*70)
        print("You only need to specify:")
        print("  • Position (X, Y, Z) in meters")
        print("  • q5 (lamp rotation) in degrees")
        print("\nThe gripper will ALWAYS point straight down!")
        print("="*70 + "\n")

    # ------------------------------------------------------------
    # Save trajectory to CSV file
    # ------------------------------------------------------------
    def save_trajectory_to_csv(self, trajectory, filename=None):
        """
        Save trajectory to CSV file with timestamp.
        Exports angles in degrees, rounded to 2 decimal places for Arduino.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Step', 'q1', 'q2', 'q3', 'q4', 'q5'])
                
                for i, q in enumerate(trajectory):
                    q_deg = np.rad2deg(q)
                    q_rounded = np.round(q_deg, 2)
                    writer.writerow([i] + q_rounded.tolist())
            
            print(f"\n✓ Trajectory saved to: {filename}")
            print(f"   Format: Step, q1°, q2°, q3°, q4°, q5°")
            self.get_logger().info(f"Trajectory saved to: {filename}")
            return True
        except Exception as e:
            print(f"\n✗ Error saving trajectory: {e}")
            self.get_logger().error(f"Error saving trajectory: {e}")
            return False

    # ------------------------------------------------------------
    # Solve IK for a given position + q5 (always pointing down)
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos_mm, q5_rad, q_init):
        """
        Solve IK for position with tool always pointing down.
        
        Args:
            ee_pos_mm: end-effector position in mm
            q5_rad: q5 rotation in radians
            q_init: initial guess
        """
        self.get_logger().info(
            f"IK Request: pos=[{ee_pos_mm[0]:.1f}, {ee_pos_mm[1]:.1f}, {ee_pos_mm[2]:.1f}] mm, "
            f"q5={np.rad2deg(q5_rad):.1f}° (tool pointing DOWN)"
        )
        
        # Try multiple initial guesses
        guesses = [
            q_init,
            np.array([0, 0, 0, 0, q5_rad]),
            np.array([0, 0.3, 0.3, 0, q5_rad]),
            np.array([0, -0.3, -0.3, 0, q5_rad]),
        ]
        
        best_error = float('inf')
        best_sol = None
        
        for i, guess in enumerate(guesses):
            q_sol, converged = self.ik.solve_ik_for_pose(ee_pos_mm, q5_rad, guess)
            
            if q_sol is None:
                continue
            
            # Verify solution
            T = self.ik.fk_node.fwd_kinematics(q_sol.tolist())
            pos_error = np.linalg.norm(T[:3, 3] - ee_pos_mm)
            R33 = self.ik.get_R33(q_sol)
            orientation_error = abs(R33 - 1.0)  # CHANGED: Should be +1.0 for pointing down
            
            combined_error = pos_error + 100.0 * orientation_error
            
            if combined_error < best_error:
                best_error = combined_error
                best_sol = q_sol
            
            # Accept good solutions early
            if pos_error < 2.0 and orientation_error < 0.05:
                self.get_logger().info(
                    f"✓ Solution found! pos_err={pos_error:.3f}mm, R33={R33:.4f}"
                )
                return q_sol, True
        
        # Check if best solution is acceptable
        if best_sol is None:
            self.get_logger().error("❌ IK FAILED - No solution found")
            return None, False
        
        T_best = self.ik.fk_node.fwd_kinematics(best_sol.tolist())
        final_pos_error = np.linalg.norm(T_best[:3, 3] - ee_pos_mm)
        final_R33 = self.ik.get_R33(best_sol)
        
        # CHANGED: Check R33 > 0.85 instead of < -0.85
        if final_pos_error < 10.0 and final_R33 > 0.85:
            self.get_logger().warn(
                f"⚠ Accepting solution: pos_err={final_pos_error:.3f}mm, R33={final_R33:.4f}"
            )
            return best_sol, True
        
        self.get_logger().error(
            f"❌ IK FAILED - pos_err={final_pos_error:.3f}mm, R33={final_R33:.4f}"
        )
        return None, False

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def run_loop(self):
        print("\n" + "="*70)
        print("POSE TRAJECTORY GENERATOR")
        print("Tool Always Pointing DOWN")
        print("="*70)
        print("\nEnter positions in meters and q5 rotation in degrees")
        
        print("\n" + "="*70)
        print("EXAMPLE SEQUENCES:")
        print("="*70)
        print("\n1. Pick lamp from table:")
        print("   Initial: 0.15 0 -0.15, q5=0")
        print("   Final:   0 0.2 -0.2, q5=0")
        print("\n2. Rotate lamp while moving:")
        print("   Initial: 0.15 0 -0.15, q5=0")
        print("   Final:   0 0.2 -0.2, q5=180")
        print("\n3. Home position test:")
        print("   Initial: 0 0 -0.404, q5=0")
        print("   Final:   0.15 0 -0.15, q5=0")
        print("="*70 + "\n")

        traj_counter = 1

        # Get first trajectory
        try:
            print("--- INITIAL POSITION ---")
            xyz0 = input("Enter INITIAL X Y Z (m): ").split()
            self.x0, self.y0, self.z0 = map(float, xyz0)
            
            q5_0_deg = float(input("Enter INITIAL q5 rotation (degrees): "))
            q5_0_rad = np.deg2rad(q5_0_deg)

            print("\n--- FINAL POSITION ---")
            xyzf = input("Enter FINAL X Y Z (m): ").split()
            self.xf, self.yf, self.zf = map(float, xyzf)
            
            q5_f_deg = float(input("Enter FINAL q5 rotation (degrees): "))
            q5_f_rad = np.deg2rad(q5_f_deg)

            print("\n--- TRAJECTORY PARAMETERS ---")
            self.Tf = float(input("Enter total movement time Tf (s): "))
            self.Ts = float(input("Enter timestep Ts (s): "))

            save_option = input("Save trajectory to CSV? (y/n): ").strip().lower()
            save_to_csv = save_option == 'y'

        except ValueError:
            print("Invalid input.")
            return

        # Convert to mm
        X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
        Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

        # Solve IK for initial position
        print("\n" + "="*70)
        print("SOLVING IK FOR INITIAL POSITION...")
        print("="*70)
        
        q_guess = np.array([0.0, 0.0, 0.0, 0.0, q5_0_rad]) if X0[2] >= 0 else np.array([0.0, -0.3, -0.3, 0.0, q5_0_rad])
        q0, conv0 = self.solve_ik_for_pose(X0, q5_0_rad, q_guess)
        
        if not conv0 or q0 is None:
            print("\n❌ IK FAILED FOR INITIAL POSITION")
            return

        print(f"\n✓ Initial joints: {np.round(np.rad2deg(q0), 1)}°")

        # Verify
        T0 = self.ik.fk_node.fwd_kinematics(q0.tolist())
        R33_0 = self.ik.get_R33(q0)
        print(f"  Achieved position: {np.round(T0[:3, 3], 1)} mm")
        print(f"  R33 (should be ~+1.0): {R33_0:.4f}")  # CHANGED: Updated comment

        # Move to initial
        msg = Float64MultiArray()
        msg.data = q0.tolist()
        self.joint_pub.publish(msg)
        print(f"\n→ Moving to initial position...")
        time.sleep(1.0)

        # Solve IK for final position
        print("\n" + "="*70)
        print("SOLVING IK FOR FINAL POSITION...")
        print("="*70)
        
        qf, convf = self.solve_ik_for_pose(Xf, q5_f_rad, q0.copy())
        
        if not convf or qf is None:
            print("\n❌ IK FAILED FOR FINAL POSITION")
            return

        print(f"\n✓ Final joints: {np.round(np.rad2deg(qf), 1)}°")

        # Verify
        Tf = self.ik.fk_node.fwd_kinematics(qf.tolist())
        R33_f = self.ik.get_R33(qf)
        print(f"  Achieved position: {np.round(Tf[:3, 3], 1)} mm")
        print(f"  R33 (should be ~+1.0): {R33_f:.4f}")  # CHANGED: Updated comment

        # Show motion summary
        print(f"\n{'='*70}")
        print("MOTION SUMMARY:")
        print(f"{'='*70}")
        print(f"Position change: {np.linalg.norm(Tf[:3, 3] - T0[:3, 3]):.1f} mm")
        print(f"q5 change: {np.rad2deg(qf[4] - q0[4]):.1f}°")
        print(f"Joint changes: {np.round(np.rad2deg(qf - q0), 1)}°")

        # Generate and execute trajectory
        trajectory = self.compute_trajectory(q0, qf, self.Tf, self.Ts)

        if save_to_csv:
            filename = f"trajectory_{traj_counter}.csv"
            self.save_trajectory_to_csv(trajectory, filename)
            traj_counter += 1

        print(f"\n{'='*70}")
        print(f"EXECUTING TRAJECTORY ({len(trajectory)} steps)")
        print(f"{'='*70}")
        for i, q in enumerate(trajectory):
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            if i % 10 == 0:
                print(f"  Step {i+1}/{len(trajectory)}")
            time.sleep(self.Ts)

        print(f"\n✓ Trajectory complete!")

        # Continue with more trajectories
        current_q = qf.copy()

        while rclpy.ok():
            print("\n" + "="*70)
            print("NEXT ACTION?")
            print("="*70)
            print("  1. New final position (from current)")
            print("  2. New initial and final positions")
            print("  3. Change trajectory timing")
            print("  q. Quit")
            
            choice = input("Choice: ").strip().lower()
            
            if choice == 'q':
                print("Exiting.")
                break
            
            elif choice == '3':
                try:
                    self.Tf = float(input(f"Total time Tf (current {self.Tf}s): "))
                    self.Ts = float(input(f"Timestep Ts (current {self.Ts}s): "))
                    print(f"✓ Updated timing")
                except:
                    print("Invalid input")
                continue
            
            elif choice == '1':
                try:
                    print("\n--- NEW FINAL POSITION ---")
                    xyzf = input("Enter FINAL X Y Z (m): ").split()
                    self.xf, self.yf, self.zf = map(float, xyzf)
                    q5_f_deg = float(input("Enter FINAL q5 (degrees): "))
                    q5_f_rad = np.deg2rad(q5_f_deg)
                except:
                    print("Invalid input")
                    continue

                Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

                print("\n" + "="*70)
                print("SOLVING IK...")
                print("="*70)
                qf, convf = self.solve_ik_for_pose(Xf, q5_f_rad, current_q.copy())
                
                if not convf or qf is None:
                    print("❌ IK FAILED")
                    continue

                trajectory = self.compute_trajectory(current_q, qf, self.Tf, self.Ts)

            elif choice == '2':
                try:
                    print("\n--- NEW INITIAL POSITION ---")
                    xyz0 = input("Enter INITIAL X Y Z (m): ").split()
                    self.x0, self.y0, self.z0 = map(float, xyz0)
                    q5_0_deg = float(input("Enter INITIAL q5 (degrees): "))
                    q5_0_rad = np.deg2rad(q5_0_deg)

                    print("\n--- NEW FINAL POSITION ---")
                    xyzf = input("Enter FINAL X Y Z (m): ").split()
                    self.xf, self.yf, self.zf = map(float, xyzf)
                    q5_f_deg = float(input("Enter FINAL q5 (degrees): "))
                    q5_f_rad = np.deg2rad(q5_f_deg)
                except:
                    print("Invalid input")
                    continue

                X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
                Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

                print("\nSolving IK for initial...")
                q0, conv0 = self.solve_ik_for_pose(X0, q5_0_rad, current_q.copy())
                if not conv0 or q0 is None:
                    print("❌ IK FAILED")
                    continue

                print("Solving IK for final...")
                qf, convf = self.solve_ik_for_pose(Xf, q5_f_rad, q0.copy())
                if not convf or qf is None:
                    print("❌ IK FAILED")
                    continue

                # Move to new initial first
                msg = Float64MultiArray()
                msg.data = q0.tolist()
                self.joint_pub.publish(msg)
                print("Moving to new initial...")
                time.sleep(1.5)

                trajectory = self.compute_trajectory(q0, qf, self.Tf, self.Ts)
                current_q = q0.copy()

            else:
                print("Invalid choice")
                continue

            # Save option
            save_option = input("\nSave trajectory? (y/n): ").strip().lower()
            if save_option == 'y':
                filename = f"trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(trajectory, filename)
                traj_counter += 1

            # Execute
            print(f"\nExecuting trajectory ({len(trajectory)} steps)...")
            for i, q in enumerate(trajectory):
                msg = Float64MultiArray()
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                if i % 10 == 0:
                    print(f"  Step {i+1}/{len(trajectory)}")
                time.sleep(self.Ts)

            print("✓ Complete!")
            current_q = qf.copy()

    # ------------------------------------------------------------
    # Cubic polynomial trajectory
    # ------------------------------------------------------------
    def compute_trajectory(self, q0, qf, Tf, Ts):
        """Generate smooth cubic trajectory in joint space"""
        n = len(q0)
        steps = int(Tf / Ts)

        qdot0 = np.zeros(n)
        qdotf = np.zeros(n)

        c0 = q0
        c1 = qdot0
        c2 = (3*(qf - q0) - (2*qdot0 + qdotf)*Tf) / (Tf**2)
        c3 = (-2*(qf - q0) + (qdot0 + qdotf)*Tf) / (Tf**3)

        traj = []
        for k in range(steps+1):
            t = k * Ts
            q_t = c0 + c1*t + c2*t**2 + c3*t**3
            traj.append(q_t)

        return traj


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrajectoryNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()