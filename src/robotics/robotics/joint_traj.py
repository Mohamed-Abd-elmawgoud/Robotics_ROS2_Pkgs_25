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

        # IK object (using full 5-DOF IK with proper weighting)
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
        
        print("\n" + "="*60)
        print("ROBOT WORKSPACE INFORMATION")
        print("="*60)
        print(f"Link lengths (mm):")
        print(f"  l1 (base height): {l1:.2f}")
        print(f"  l2 (upper arm):   {l2:.2f}")
        print(f"  l3 (forearm):     {l3:.2f}")
        print(f"  l4 + l5 (tool):   {l4 + l5:.2f}")
        print(f"\nApproximate reach:")
        print(f"  Max horizontal: ~{l2 + l3:.1f} mm = {(l2+l3)/1000:.3f} m")
        print(f"  Max vertical (up): ~{l2 + l3 - l1:.1f} mm")
        print(f"  Max vertical (down): ~{-(l2 + l3 + l1):.1f} mm")
        print(f"  Tool length: {l4 + l5:.1f} mm (subtracted to get wrist center)")
        print(f"\nIK Method: Decoupled 3+1+1 DOF")
        print(f"  - 3-DOF position IK for q1, q2, q3")
        print(f"  - q4 solved for pitch (R33 control)")
        print(f"  - q5 set directly (tool rotation)")
        print(f"\nPitch convention:")
        print(f"  pitch = -90° → R33 = -1 (tool pointing DOWN)")
        print(f"  pitch = 0°   → R33 = 0  (tool HORIZONTAL)")
        print(f"  pitch = +90° → R33 = 1  (tool pointing UP)")
        print("="*60 + "\n")

    # ------------------------------------------------------------
    # Save trajectory to CSV file
    # ------------------------------------------------------------
    def save_trajectory_to_csv(self, trajectory, filename=None):
        """
        Save trajectory to CSV file with timestamp.
        Exports angles in degrees, rounded to 2 decimal places for Arduino.
        
        Args:
            trajectory: list of numpy arrays (joint angles)
            filename: optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header (Arduino-friendly format)
                writer.writerow(['Step', 'q1', 'q2', 'q3', 'q4', 'q5'])
                
                # Write data - angles in degrees, rounded to 2 decimal places
                for i, q in enumerate(trajectory):
                    q_deg = np.rad2deg(q)
                    q_rounded = np.round(q_deg, 2)
                    writer.writerow([i] + q_rounded.tolist())
            
            print(f"\n✓ Trajectory saved to: {filename}")
            print(f"   Format: Step, q1°, q2°, q3°, q4°, q5° (degrees, rounded to 2 decimals)")
            self.get_logger().info(f"Trajectory saved to: {filename}")
            return True
        except Exception as e:
            print(f"\n✗ Error saving trajectory: {e}")
            self.get_logger().error(f"Error saving trajectory: {e}")
            return False

    # ------------------------------------------------------------
    # Solve IK for a given pose (position + pitch + q5)
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos_mm, pitch_rad, q5_rad, q_init):
        """
        Solve IK using decoupled approach.
        Now uses 3-DOF position IK + q4 for pitch + q5 for rotation.
        """
        self.get_logger().info(
            f"Target: EE=[{ee_pos_mm[0]:.1f}, {ee_pos_mm[1]:.1f}, {ee_pos_mm[2]:.1f}] mm, "
            f"pitch={np.rad2deg(pitch_rad):.1f}°, q5={np.rad2deg(q5_rad):.1f}°"
        )
        
        # Try multiple initial guesses for robustness
        guesses = [
            q_init,
            np.array([0, 0, 0, 0, q5_rad]),
            np.array([0, 0.3, 0.3, 0, q5_rad]),
            np.array([0, -0.3, -0.3, 0, q5_rad]),
        ]
        
        best_error = float('inf')
        best_sol = None
        best_converged = False
        
        for i, guess in enumerate(guesses):
            q_sol, converged = self.ik.solve_ik_for_pose(ee_pos_mm, pitch_rad, q5_rad, guess)
            
            if q_sol is None:
                continue
            
            # Verify solution quality
            T = self.ik.fk_node.fwd_kinematics(q_sol.tolist())
            pos_error = np.linalg.norm(T[:3, 3] - ee_pos_mm)
            
            # Check pitch error using R33
            R33_current = self.ik.get_R33(q_sol)
            R33_target = self.ik.pitch_to_R33(pitch_rad)
            R33_error = abs(R33_current - R33_target)
            
            # Also get pitch angle for reporting
            current_pitch = self.ik.get_current_pitch(q_sol)
            pitch_error = abs(pitch_rad - current_pitch)
            pitch_error = abs(np.arctan2(np.sin(pitch_error), np.cos(pitch_error)))
            
            # Combined error
            combined_error = pos_error + 100.0 * R33_error
            
            if combined_error < best_error:
                best_error = combined_error
                best_sol = q_sol
                best_converged = converged
            
            # If we found a good solution, use it
            if pos_error < 2.0 and R33_error < 0.05:
                self.get_logger().info(
                    f"✓ Solution found (guess {i})! "
                    f"pos_err={pos_error:.3f}mm, pitch_err={np.rad2deg(pitch_error):.2f}°, R33_err={R33_error:.4f}"
                )
                print(f"  → Joint angles (deg): {np.round(np.rad2deg(q_sol), 2)}")
                print(f"  → R33: {R33_current:.4f} (target: {R33_target:.4f})")
                return q_sol, True
        
        # Check if best solution is acceptable
        if best_sol is None:
            self.get_logger().error("IK FAILED! No solution found from any initial guess.")
            return None, False
        
        # Verify best solution
        T_best = self.ik.fk_node.fwd_kinematics(best_sol.tolist())
        final_pos_error = np.linalg.norm(T_best[:3, 3] - ee_pos_mm)
        
        R33_best = self.ik.get_R33(best_sol)
        R33_target = self.ik.pitch_to_R33(pitch_rad)
        R33_error = abs(R33_best - R33_target)
        
        final_pitch = self.ik.get_current_pitch(best_sol)
        final_pitch_error = abs(pitch_rad - final_pitch)
        final_pitch_error = abs(np.arctan2(np.sin(final_pitch_error), np.cos(final_pitch_error)))
        
        if final_pos_error < 10.0 and R33_error < 0.1:
            self.get_logger().warn(
                f"Accepting best solution: pos_err={final_pos_error:.3f}mm, "
                f"pitch_err={np.rad2deg(final_pitch_error):.2f}°, R33_err={R33_error:.4f}"
            )
            print(f"  → Joint angles (deg): {np.round(np.rad2deg(best_sol), 2)}")
            print(f"  → R33: {R33_best:.4f} (target: {R33_target:.4f})")
            return best_sol, True
        
        self.get_logger().error(
            f"IK FAILED! Best: pos_err={final_pos_error:.3f}mm, "
            f"pitch_err={np.rad2deg(final_pitch_error):.2f}°, R33_err={R33_error:.4f}"
        )
        return None, False

    # ------------------------------------------------------------
    # Main loop: ask for input, generate trajectory, publish
    # ------------------------------------------------------------
    def run_loop(self):
        print("\n=== Pose Trajectory Generator (Position + Orientation) ===")
        print("Enter positions in meters and angles in degrees")
        
        # Show example lamp pick-and-place sequence
        print("\n" + "="*70)
        print("EXAMPLE LAMP PICK-AND-PLACE SEQUENCE:")
        print("="*70)
        print("Lamp on table (pick):")
        print("  Position: 0.15 0 -0.15 (150mm forward, 150mm below base)")
        print("  Pitch: -90 (gripper pointing down)")
        print("  q5: 0")
        print("\nSocket location (place):")
        print("  Position: 0 0.25 -0.2 (250mm to side)")
        print("  Pitch: 90 (gripper pointing up to screw in)")
        print("  q5: 180 (rotate lamp 180° while moving)")
        print("\nHome position:")
        print("  Position: 0 0 -0.404 (straight down from base)")
        print("  Pitch: -90 (pointing down)")
        print("  q5: 0")
        print("\nTest orientation change (same position):")
        print("  Position: 0.15 0 -0.15")
        print("  Pitch: 0 (gripper horizontal)")
        print("  q5: 0")
        print("="*70 + "\n")

        # Counter for trajectory files
        traj_counter = 1

        # -----------------------------------------
        # First input: initial and final pose
        # -----------------------------------------
        try:
            print("\n--- INITIAL POSE ---")
            xyz0 = input("Enter INITIAL X Y Z (m): ").split()
            self.x0, self.y0, self.z0 = map(float, xyz0)
            
            pitch0_deg = float(input("Enter INITIAL pitch angle (degrees): "))
            pitch0_rad = np.deg2rad(pitch0_deg)
            
            q5_0_deg = float(input("Enter INITIAL q5/lamp rotation (degrees): "))
            q5_0_rad = np.deg2rad(q5_0_deg)

            print("\n--- FINAL POSE ---")
            xyzf = input("Enter FINAL X Y Z (m): ").split()
            self.xf, self.yf, self.zf = map(float, xyzf)
            
            pitch_f_deg = float(input("Enter FINAL pitch angle (degrees): "))
            pitch_f_rad = np.deg2rad(pitch_f_deg)
            
            q5_f_deg = float(input("Enter FINAL q5/lamp rotation (degrees): "))
            q5_f_rad = np.deg2rad(q5_f_deg)

            print("\n--- TRAJECTORY PARAMETERS ---")
            self.Tf = float(input("Enter total movement time Tf (s): "))
            self.Ts = float(input("Enter timestep Ts (s): "))

            save_option = input("Save trajectory to CSV? (y/n): ").strip().lower()
            save_to_csv = save_option == 'y'

        except ValueError:
            print("Invalid input, try again.")
            return

        # Convert positions to mm
        X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
        Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

        # Solve IK for initial pose
        print("\n" + "="*50)
        print("SOLVING IK FOR INITIAL POSE...")
        print("="*50)
        print(f"Position (mm): [{X0[0]:.1f}, {X0[1]:.1f}, {X0[2]:.1f}]")
        print(f"Pitch: {pitch0_deg:.1f}°, q5: {q5_0_deg:.1f}°")
        
        # Better initial guess based on typical configuration
        if X0[2] < 0:
            q_guess = np.array([0.0, -0.3, -0.3, 0.0, q5_0_rad])
        else:
            q_guess = np.array([0.0, 0.3, 0.3, 0.0, q5_0_rad])
            
        q0, conv0 = self.solve_ik_for_pose(X0, pitch0_rad, q5_0_rad, q_guess)
        
        if not conv0:
            print("\n❌ IK FAILED FOR INITIAL POSE!")
            return

        print(f"✓ Initial joints (deg): {np.round(np.rad2deg(q0), 2)}")

        # Verify initial pose
        T0 = self.ik.fk_node.fwd_kinematics(q0.tolist())
        actual_pitch0 = self.ik.get_current_pitch(q0)
        R33_0 = self.ik.get_R33(q0)
        print(f"  Achieved position: {np.round(T0[:3, 3], 2)} mm")
        print(f"  Achieved pitch: {np.rad2deg(actual_pitch0):.1f}°")
        print(f"  Achieved R33: {R33_0:.4f}")

        # Move to initial position
        msg = Float64MultiArray()
        msg.data = q0.tolist()
        self.joint_pub.publish(msg)
        self.get_logger().info(f"Moving to initial position...")
        print(f"→ Publishing to 'joint_angles_out' topic")
        time.sleep(1.0)

        # Solve IK for final pose
        print("\n" + "="*50)
        print("SOLVING IK FOR FINAL POSE...")
        print("="*50)
        print(f"Position (mm): [{Xf[0]:.1f}, {Xf[1]:.1f}, {Xf[2]:.1f}]")
        print(f"Pitch: {pitch_f_deg:.1f}°, q5: {q5_f_deg:.1f}°")
        
        qf, convf = self.solve_ik_for_pose(Xf, pitch_f_rad, q5_f_rad, q0.copy())
        
        if not convf:
            print("\n❌ IK FAILED FOR FINAL POSE!")
            return

        print(f"✓ Final joints (deg): {np.round(np.rad2deg(qf), 2)}")

        # Verify final pose
        Tf = self.ik.fk_node.fwd_kinematics(qf.tolist())
        actual_pitchf = self.ik.get_current_pitch(qf)
        R33_f = self.ik.get_R33(qf)
        print(f"  Achieved position: {np.round(Tf[:3, 3], 2)} mm")
        print(f"  Achieved pitch: {np.rad2deg(actual_pitchf):.1f}°")
        print(f"  Achieved R33: {R33_f:.4f}")

        # Show changes
        print(f"\n{'='*50}")
        print("TRAJECTORY CHANGES:")
        print(f"{'='*50}")
        print(f"Position change: {np.linalg.norm(Tf[:3, 3] - T0[:3, 3]):.1f} mm")
        print(f"Pitch change: {np.rad2deg(actual_pitchf - actual_pitch0):.1f}°")
        print(f"Joint changes (deg): {np.round(np.rad2deg(qf - q0), 2)}")

        # Generate trajectory
        trajectory = self.compute_trajectory(q0, qf, self.Tf, self.Ts)

        # Save first trajectory if requested
        if save_to_csv:
            filename = f"trajectory_{traj_counter}.csv"
            self.save_trajectory_to_csv(trajectory, filename)
            traj_counter += 1

        # Execute first trajectory
        print(f"\n{'='*50}")
        print(f"EXECUTING TRAJECTORY ({len(trajectory)} steps)")
        print(f"{'='*50}")
        for i, q in enumerate(trajectory):
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            if i % 10 == 0:  # Print every 10th step
                print(f"  Step {i+1}/{len(trajectory)}: {np.round(np.rad2deg(q), 1)}°")
            time.sleep(self.Ts)

        print(f"✓ Trajectory complete!")

        # -----------------------------------------
        # Subsequent movements
        # -----------------------------------------
        current_q = qf.copy()

        while rclpy.ok():
            print("\n" + "="*50)
            print("TRAJECTORY COMPLETE - NEXT ACTION?")
            print("="*50)
            print("Options:")
            print("  1. Enter new FINAL pose (from current position)")
            print("  2. Enter new INITIAL and FINAL poses")
            print("  3. Change trajectory parameters (Tf, Ts)")
            print("  q. Quit")
            
            choice = input("Enter choice: ").strip().lower()
            
            if choice == 'q':
                print("Exiting.")
                break
            
            elif choice == '3':
                # Change trajectory parameters
                try:
                    print("\n--- TRAJECTORY PARAMETERS ---")
                    self.Tf = float(input(f"Enter total movement time Tf (current: {self.Tf}s): "))
                    self.Ts = float(input(f"Enter timestep Ts (current: {self.Ts}s): "))
                    print(f"✓ Updated: Tf={self.Tf}s, Ts={self.Ts}s")
                except ValueError:
                    print("Invalid input, keeping current values.")
                continue
            
            elif choice == '1':
                # Use current position as initial
                try:
                    print("\n--- NEW FINAL POSE ---")
                    xyzf = input("Enter FINAL X Y Z (m): ").split()
                    self.xf, self.yf, self.zf = map(float, xyzf)
                    
                    pitch_f_deg = float(input("Enter FINAL pitch angle (degrees): "))
                    pitch_f_rad = np.deg2rad(pitch_f_deg)
                    
                    q5_f_deg = float(input("Enter FINAL q5/lamp rotation (degrees): "))
                    q5_f_rad = np.deg2rad(q5_f_deg)
                    
                except ValueError:
                    print("Invalid input, try again.")
                    continue

                Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

                # Show current state
                T_current = self.ik.fk_node.fwd_kinematics(current_q.tolist())
                current_pitch = self.ik.get_current_pitch(current_q)
                current_R33 = self.ik.get_R33(current_q)
                print(f"\nCurrent position: {np.round(T_current[:3, 3], 2)} mm")
                print(f"Current pitch: {np.rad2deg(current_pitch):.1f}°")
                print(f"Current R33: {current_R33:.4f}")
                print(f"Current joints: {np.round(np.rad2deg(current_q), 1)}°")

                # Solve IK for new final pose
                print("\n" + "="*50)
                print("SOLVING IK FOR NEW FINAL POSE...")
                print("="*50)
                print(f"Target position (mm): [{Xf[0]:.1f}, {Xf[1]:.1f}, {Xf[2]:.1f}]")
                print(f"Target pitch: {pitch_f_deg:.1f}°, q5: {q5_f_deg:.1f}°")
                
                qf, convf = self.solve_ik_for_pose(Xf, pitch_f_rad, q5_f_rad, current_q.copy())
                
                if not convf:
                    print("\n❌ IK FAILED FOR FINAL POSE!")
                    continue

                print(f"✓ New final joints (deg): {np.round(np.rad2deg(qf), 2)}")

                # Verify and show changes
                Tf = self.ik.fk_node.fwd_kinematics(qf.tolist())
                actual_pitchf = self.ik.get_current_pitch(qf)
                R33_f = self.ik.get_R33(qf)
                print(f"  Achieved position: {np.round(Tf[:3, 3], 2)} mm")
                print(f"  Achieved pitch: {np.rad2deg(actual_pitchf):.1f}°")
                print(f"  Achieved R33: {R33_f:.4f}")
                print(f"\nChanges:")
                print(f"  Position: {np.linalg.norm(Tf[:3, 3] - T_current[:3, 3]):.1f} mm")
                print(f"  Pitch: {np.rad2deg(actual_pitchf - current_pitch):.1f}°")
                print(f"  R33: {R33_f - current_R33:.4f}")
                print(f"  Joints: {np.round(np.rad2deg(qf - current_q), 2)}°")

                # Generate trajectory from current to new final
                trajectory = self.compute_trajectory(current_q, qf, self.Tf, self.Ts)

            elif choice == '2':
                # Get both new initial and final poses
                try:
                    print("\n--- NEW INITIAL POSE ---")
                    xyz0 = input("Enter INITIAL X Y Z (m): ").split()
                    self.x0, self.y0, self.z0 = map(float, xyz0)
                    
                    pitch0_deg = float(input("Enter INITIAL pitch angle (degrees): "))
                    pitch0_rad = np.deg2rad(pitch0_deg)
                    
                    q5_0_deg = float(input("Enter INITIAL q5/lamp rotation (degrees): "))
                    q5_0_rad = np.deg2rad(q5_0_deg)

                    print("\n--- NEW FINAL POSE ---")
                    xyzf = input("Enter FINAL X Y Z (m): ").split()
                    self.xf, self.yf, self.zf = map(float, xyzf)
                    
                    pitch_f_deg = float(input("Enter FINAL pitch angle (degrees): "))
                    pitch_f_rad = np.deg2rad(pitch_f_deg)
                    
                    q5_f_deg = float(input("Enter FINAL q5/lamp rotation (degrees): "))
                    q5_f_rad = np.deg2rad(q5_f_deg)
                    
                except ValueError:
                    print("Invalid input, try again.")
                    continue

                X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
                Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

                # Solve IK for both poses
                print("\nSOLVING IK FOR INITIAL POSE...")
                q0, conv0 = self.solve_ik_for_pose(X0, pitch0_rad, q5_0_rad, current_q.copy())
                
                if not conv0:
                    print("❌ IK FAILED FOR INITIAL POSE!")
                    continue

                print("SOLVING IK FOR FINAL POSE...")
                qf, convf = self.solve_ik_for_pose(Xf, pitch_f_rad, q5_f_rad, q0.copy())
                
                if not convf:
                    print("❌ IK FAILED FOR FINAL POSE!")
                    continue

                # Move to new initial position first
                msg = Float64MultiArray()
                msg.data = q0.tolist()
                self.joint_pub.publish(msg)
                print(f"\n→ Moving to new initial position...")
                time.sleep(1.5)

                trajectory = self.compute_trajectory(q0, qf, self.Tf, self.Ts)
                current_q = q0.copy()

            else:
                print("Invalid choice.")
                continue

            # Ask about CSV for each trajectory
            save_option = input("\nSave this trajectory to CSV? (y/n): ").strip().lower()
            if save_option == 'y':
                filename = f"trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(trajectory, filename)
                traj_counter += 1

            # Execute trajectory
            print(f"\n{'='*50}")
            print(f"EXECUTING TRAJECTORY ({len(trajectory)} steps)")
            print(f"{'='*50}")
            for i, q in enumerate(trajectory):
                msg = Float64MultiArray()
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                if i % 10 == 0:
                    print(f"  Step {i+1}/{len(trajectory)}: {np.round(np.rad2deg(q), 1)}°")
                time.sleep(self.Ts)

            print(f"✓ Trajectory complete!")
            current_q = qf.copy()

    # ------------------------------------------------------------
    # Cubic polynomial trajectory
    # ------------------------------------------------------------
    def compute_trajectory(self, q0, qf, Tf, Ts):
        """
        Generate smooth cubic polynomial trajectory in joint space
        All 5 joints (including q4 and q5) are interpolated smoothly
        """
        n = len(q0)
        steps = int(Tf / Ts)

        # Zero velocity at start and end
        qdot0 = np.zeros(n)
        qdotf = np.zeros(n)

        # Cubic polynomial coefficients
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