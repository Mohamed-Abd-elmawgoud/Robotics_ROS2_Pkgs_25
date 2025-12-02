#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import csv
from datetime import datetime
from robotics.inverse_kinematics import InverseKinematicsNode


class JointTrajectoryNode(Node):
    def __init__(self):
        super().__init__('joint_space_trajectory')

        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)

        # IK object (used for solving q0 and qf)
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
        print("\n=== Joint Space Trajectory Generator ===")

        # Counter for trajectory files
        traj_counter = 1

        # -----------------------------------------
        # First input: initial and final pose
        # -----------------------------------------
        try:
            xyz0 = input("Enter INITIAL X Y Z (m): ").split()
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

        # Convert to IK units
        X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
        Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

        # Solve IK for initial & final pose
        print("\nSolving inverse kinematics for initial pose...")
        q_guess = np.zeros_like(self.ik.q_current)
        q0, conv0 = self.ik.inv_kinematics(q_guess, X0)
        if not conv0:
            print("IK failed for initial point.")
            return

        msg = Float64MultiArray()
        msg.data = q0.tolist()
        self.joint_pub.publish(msg)
        self.get_logger().info(f"Moved to q0: {np.round(q0,4)}")
        time.sleep(0.5)

        print("Solving IK for final pose...")
        qf, convf = self.ik.inv_kinematics(q0.copy(), Xf)
        if not convf:
            print("IK failed for final point.")
            return

        print("\nIK successful.")
        trajectory = self.compute_trajectory(q0, qf, Tf, Ts)

        # Save first trajectory if requested
        if save_to_csv:
            filename = f"trajectory_{traj_counter}.csv"
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
        current_q = qf.copy()  # start next movement from last final position
        while rclpy.ok():
            print("\nTrajectory complete. Enter new FINAL position (X Y Z) or 'q' to quit:")

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

            # Solve IK for new final pose
            qf, convf = self.ik.inv_kinematics(current_q.copy(), Xf)
            if not convf:
                print("IK failed for final point.")
                continue

            print("Generating trajectory...")
            trajectory = self.compute_trajectory(current_q, qf, Tf, Ts)

            # Save trajectory if requested
            if save_to_csv:
                filename = f"trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(trajectory, filename)
                traj_counter += 1

            # Execute trajectory
            for i, q in enumerate(trajectory):
                msg = Float64MultiArray()
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                self.get_logger().info(f"Step {i+1}/{len(trajectory)}: {np.round(q,3)}")
                time.sleep(Ts)

            current_q = qf.copy()


    # ------------------------------------------------------------
    # Cubic polynomial trajectory
    # ------------------------------------------------------------
    def compute_trajectory(self, q0, qf, Tf, Ts):
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
    node = JointTrajectoryNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()