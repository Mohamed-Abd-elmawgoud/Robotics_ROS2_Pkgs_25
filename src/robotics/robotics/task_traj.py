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

        # IK object
        self.ik = InverseKinematicsNode()

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
    # Main loop: ask for input, generate trajectory, publish
    # ------------------------------------------------------------
    def run_loop(self):
        print("\n=== Task Space Trajectory Generator ===")
        print("(End effector will move in a STRAIGHT LINE between points)")

        # Counter for trajectory files
        traj_counter = 1

        # -----------------------------------------
        # First input: initial and final pose
        # -----------------------------------------
        try:
            xyz0 = input("\nEnter INITIAL X Y Z (m): ").split()
            self.x0, self.y0, self.z0 = map(float, xyz0)

            xyzf = input("Enter FINAL X Y Z (m): ").split()
            self.xf, self.yf, self.zf = map(float, xyzf)

            Tf = float(input("Enter total movement time Tf (s): "))
            Ts = float(input("Enter timestep Ts (s): "))

            save_option = input("Save trajectory to CSV? (y/n): ").strip().lower()
            save_to_csv = save_option == 'y'

        except ValueError:
            print("Invalid input, try again.")
            return

        # Convert to IK units (mm)
        X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
        Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

        # Solve IK for initial pose
        print("\nSolving inverse kinematics for initial pose...")
        q_guess = np.zeros_like(self.ik.q_current)
        q0, conv0 = self.ik.inv_kinematics(q_guess, X0)
        if not conv0:
            print("IK failed for initial point.")
            return

        # Move to initial position
        msg = Float64MultiArray()
        msg.data = q0.tolist()
        self.joint_pub.publish(msg)
        self.get_logger().info(f"Moved to initial position q0: {np.round(q0,4)}")
        time.sleep(0.5)

        # Generate and execute trajectory
        print(f"\nGenerating straight-line trajectory from {X0/1000} to {Xf/1000} m...")
        trajectory, success = self.compute_task_space_trajectory(q0, X0, Xf, Tf, Ts)
        
        if not success:
            print("Failed to generate trajectory. IK may have failed at some waypoints.")
            return

        print(f"✓ Trajectory generated: {len(trajectory)} steps")

        # Save first trajectory if requested
        if save_to_csv:
            filename = f"task_trajectory_{traj_counter}.csv"
            self.save_trajectory_to_csv(trajectory, filename)
            traj_counter += 1

        # Execute first trajectory
        for i, q in enumerate(trajectory):
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            self.get_logger().info(f"Step {i+1}/{len(trajectory)}: {np.round(q,3)}")
            time.sleep(Ts)

        # -----------------------------------------
        # Subsequent movements
        # -----------------------------------------
        current_X = Xf.copy()  # Current end effector position
        current_q = trajectory[-1].copy()  # Last joint configuration
        
        while rclpy.ok():
            print("\n" + "="*50)
            print("Trajectory complete. Enter new FINAL position or 'q' to quit:")

            xyzf = input("Enter FINAL X Y Z (m): ").strip()
            if xyzf.lower() == 'q':
                print("Exiting.")
                break

            try:
                self.xf, self.yf, self.zf = map(float, xyzf.split())
            except ValueError:
                print("Invalid input, try again.")
                continue

            Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

            # Generate straight-line trajectory from current position to new final position
            print(f"\nGenerating straight-line trajectory from {current_X/1000} to {Xf/1000} m...")
            trajectory, success = self.compute_task_space_trajectory(current_q, current_X, Xf, Tf, Ts)
            
            if not success:
                print("Failed to generate trajectory. Try a different target position.")
                continue

            print(f"✓ Trajectory generated: {len(trajectory)} steps")

            # Save trajectory if requested
            if save_to_csv:
                filename = f"task_trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(trajectory, filename)
                traj_counter += 1

            # Execute trajectory
            for i, q in enumerate(trajectory):
                msg = Float64MultiArray()
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                self.get_logger().info(f"Step {i+1}/{len(trajectory)}: {np.round(q,3)}")
                time.sleep(Ts)

            # Update current state
            current_X = Xf.copy()
            current_q = trajectory[-1].copy()

    # ------------------------------------------------------------
    # Task Space Trajectory: Linear interpolation in Cartesian space
    # ------------------------------------------------------------
    def compute_task_space_trajectory(self, q_initial, X0, Xf, Tf, Ts):
        """
        Generate trajectory with straight-line motion in task space.
        
        Args:
            q_initial: initial joint configuration (for IK seed)
            X0: initial position [x, y, z] in mm
            Xf: final position [x, y, z] in mm
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
            
            # Solve IK for this waypoint
            q_waypoint, converged = self.ik.inv_kinematics(q_current.copy(), X_waypoint)
            
            if not converged:
                print(f"✗ IK failed at step {k}/{steps} (waypoint: {X_waypoint/1000} m)")
                return trajectory, False
            
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