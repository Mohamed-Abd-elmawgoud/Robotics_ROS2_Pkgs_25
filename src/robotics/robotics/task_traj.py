#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import csv
from datetime import datetime
from robotics.inverse_kinematics import InverseKinematicsNode


class TaskSpaceTrajectoryNode(Node):
    def __init__(self):
        super().__init__('task_space_trajectory')

        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)

        # IK object (always pointing down mode)
        self.ik = InverseKinematicsNode()
        
        # Default trajectory parameters
        self.Tf = 5.0  # seconds
        self.Ts = 0.1  # seconds
        
        # Track current joint configuration
        self.current_q = np.zeros(5)
        self.current_q5 = 0.0
        self.current_position = None

        # Start main loop
        self.run_loop()

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
    # Solve IK using the new symbolic full IK solver
    # ------------------------------------------------------------
    def solve_ik_for_pose(self, ee_pos_mm, q5_rad, q_init):
        """
        Solve IK using the new solve_ik_full method (position + q5, always pointing down).
        """
        self.get_logger().info(
            f"IK Request: pos=[{ee_pos_mm[0]:.1f}, {ee_pos_mm[1]:.1f}, {ee_pos_mm[2]:.1f}] mm, "
            f"q5={np.rad2deg(q5_rad):.1f}°"
        )

        # Call new IK function
        q_sol, success = self.ik.solve_ik_full(q_init, ee_pos_mm, q5_rad)
        
        if not success:
            self.get_logger().error("❌ IK FAILED - No solution found")
            return None, False

        # Flatten joint solution
        if isinstance(q_sol, tuple):
            q_flat = np.hstack([np.atleast_1d(x) for x in q_sol])
        else:
            q_flat = np.array(q_sol)
        
        # Ensure we only have 5 joints
        q_flat = q_flat[:5]

        # Verify solution
        T = self.ik.fk_node.fwd_kinematics(q_flat.tolist())
        pos_error = np.linalg.norm(T[:3, 3] - ee_pos_mm)
        R33 = self.ik.get_R33(q_flat)
        
        if pos_error > 10.0 or R33 < 0.85:
            self.get_logger().warn(
                f"⚠ Accepting IK solution with errors: pos_err={pos_error:.3f} mm, R33={R33:.4f}"
            )

        return q_flat, True

    # ------------------------------------------------------------
    # Main loop: ask for input, generate trajectory, publish
    # ------------------------------------------------------------
    def run_loop(self):
        print("\n" + "="*70)
        print("TASK SPACE TRAJECTORY GENERATOR")
        print("Tool Always Pointing DOWN - Straight Line Motion in Cartesian Space")
        print("="*70)
        print("\nEnter positions in meters and q5 rotation in degrees")
        print("Type 'quit' or 'q' to exit\n")

        # Counter for trajectory files
        traj_counter = 1
        first_trajectory = True

        while True:
            try:
                print("\n" + "-"*70)
                print(f"TRAJECTORY #{traj_counter}")
                print("-"*70)
                
                # For first trajectory, ask for initial position
                if first_trajectory:
                    user_input = input("Enter INITIAL X Y Z (m): ").strip()
                    if user_input.lower() in ['quit', 'q']:
                        break
                    xyz0 = np.array(list(map(float, user_input.split())))
                    
                    user_input = input("Enter INITIAL q5 rotation (degrees): ").strip()
                    if user_input.lower() in ['quit', 'q']:
                        break
                    q5_0_rad = np.deg2rad(float(user_input))
                    
                    X0 = xyz0 * 1000  # Convert to mm
                else:
                    # Use previous final configuration as initial
                    print(f"Using previous final position as initial:")
                    print(f"  Position: [{self.current_position[0]/1000:.3f}, {self.current_position[1]/1000:.3f}, {self.current_position[2]/1000:.3f}] m")
                    print(f"  q5: {np.rad2deg(self.current_q5):.1f}°")
                    X0 = self.current_position.copy()
                    q5_0_rad = self.current_q5

                # Ask for final position
                user_input = input("Enter FINAL X Y Z (m): ").strip()
                if user_input.lower() in ['quit', 'q']:
                    break
                xyzf = np.array(list(map(float, user_input.split())))
                
                user_input = input("Enter FINAL q5 rotation (degrees): ").strip()
                if user_input.lower() in ['quit', 'q']:
                    break
                q5_f_rad = np.deg2rad(float(user_input))

                user_input = input("Enter total movement time Tf (s): ").strip()
                if user_input.lower() in ['quit', 'q']:
                    break
                self.Tf = float(user_input)
                
                user_input = input("Enter timestep Ts (s): ").strip()
                if user_input.lower() in ['quit', 'q']:
                    break
                self.Ts = float(user_input)
                
                save_to_csv = input("Save trajectory to CSV? (y/n): ").strip().lower() == 'y'

            except Exception as e:
                print(f"Invalid input: {e}")
                print("Please try again or type 'quit' to exit.")
                continue

            # Convert final position to mm
            Xf = xyzf * 1000

            # Solve IK for initial position
            if first_trajectory:
                q_guess = np.zeros(5)
                q0, conv0 = self.solve_ik_for_pose(X0, q5_0_rad, q_guess)
                if not conv0 or q0 is None:
                    print("❌ IK FAILED FOR INITIAL POSITION")
                    print("Please try different values.")
                    continue
                # Set q4 to 0 for initial configuration
                q0[3] = 0
            else:
                # Use previous final joints as initial
                q0 = self.current_q.copy()

            print(f"\n✓ Initial joints: {np.round(np.rad2deg(q0), 1)}°")
            
            # Generate straight-line trajectory from current position to new final position
            print(f"\nGenerating straight-line trajectory from {X0/1000} to {Xf/1000} m...")
            trajectory, success = self.compute_task_space_trajectory(q0, X0, Xf, q5_0_rad, q5_f_rad, self.Tf, self.Ts)
            
            if not success:
                print("Failed to generate trajectory. Try a different target position.")
                continue

            print(f"✓ Trajectory generated: {len(trajectory)} steps")
            print(f"✓ Final joints:   {np.round(np.rad2deg(trajectory[-1]), 1)}°")

            # Save trajectory if requested
            if save_to_csv:
                filename = f"task_trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(trajectory, filename)

            # Execute trajectory
            print(f"\n▶ Executing trajectory {traj_counter}...")
            msg = Float64MultiArray()
            for i, q in enumerate(trajectory):
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                if i % 10 == 0:
                    print(f"  Step {i+1}/{len(trajectory)}")
                time.sleep(self.Ts)
            
            print(f"✓ Trajectory {traj_counter} complete!")

            # Update current configuration for next trajectory
            self.current_q = trajectory[-1].copy()
            self.current_q5 = q5_f_rad
            self.current_position = Xf.copy()
            
            # Update counters
            traj_counter += 1
            first_trajectory = False
            
            # Ask if user wants to continue
            continue_input = input("\nGenerate another trajectory? (y/n): ").strip().lower()
            if continue_input not in ['y', 'yes']:
                break

        print("\n" + "="*70)
        print("Trajectory generation complete. Exiting...")
        print("="*70)

    # ------------------------------------------------------------
    # Task Space Trajectory: Linear interpolation in Cartesian space
    # ------------------------------------------------------------
    def compute_task_space_trajectory(self, q_initial, X0, Xf, q5_0, q5_f, Tf, Ts):
        """
        Generate trajectory with straight-line motion in task space.
        Tool is constrained to point DOWN throughout the motion.
        
        Args:
            q_initial: initial joint configuration (for IK seed)
            X0: initial position [x, y, z] in mm
            Xf: final position [x, y, z] in mm
            q5_0: initial q5 angle in radians
            q5_f: final q5 angle in radians
            Tf: total time
            Ts: time step
            
        Returns:
            trajectory: list of joint angle arrays
            success: True if all IK solutions found
        """
        steps = int(Tf / Ts)
        trajectory = []
        
        # Current joint config for warm-starting IK
        q_current = q_initial.copy()
        
        # Generate waypoints along straight line
        for k in range(steps + 1):
            # Linear interpolation parameter (0 to 1)
            s = k / steps if steps > 0 else 0
            
            # Cartesian waypoint (straight line)
            X_waypoint = X0 + s * (Xf - X0)
            
            # Interpolate q5 rotation
            q5_waypoint = q5_0 + s * (q5_f - q5_0)
            
            # Solve IK for this waypoint
            q_waypoint, converged = self.solve_ik_for_pose(X_waypoint, q5_waypoint, q_current.copy())
            
            if not converged or q_waypoint is None:
                print(f"✗ IK failed at step {k}/{steps} (waypoint: {X_waypoint/1000} m)")
                return trajectory, False
            
            # ENFORCE q4 to keep tool pointing DOWN
            # For vertical pick-up motion, we want q4 to stay close to its initial value
            # This maintains the downward pointing constraint
            if k == 0:
                # Store the initial q4 value
                self.q4_constraint = q_waypoint[3]
            else:
                # Keep q4 constant throughout the trajectory
                q_waypoint[3] = self.q4_constraint
            
            trajectory.append(q_waypoint)
            
            # Update seed for next IK (warm start for smooth motion)
            q_current = q_waypoint.copy()
        
        return trajectory, True


def main(args=None):
    rclpy.init(args=args)
    node = TaskSpaceTrajectoryNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()