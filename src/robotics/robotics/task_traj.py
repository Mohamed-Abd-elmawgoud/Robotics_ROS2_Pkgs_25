#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import csv
from datetime import datetime

# Try importing IK node
try:
    from robotics.inverse_kinematics import InverseKinematicsNode
except ImportError:
    try:
        from inverse_kinematics import InverseKinematicsNode
    except ImportError:
        print("Warning: Could not import InverseKinematicsNode")
        InverseKinematicsNode = None


class TaskSpaceTrajectoryNode(Node):
    def __init__(self):
        super().__init__('task_space_trajectory')

        # Publisher for joint angles
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_angles_out', 10)

        # IK object (used for solving inverse kinematics)
        if InverseKinematicsNode is None:
            self.get_logger().error("InverseKinematicsNode not available!")
            return
        self.ik = InverseKinematicsNode()

        # Start main loop
        self.run_loop()

    # ------------------------------------------------------------
    # Save trajectory to CSV file
    # ------------------------------------------------------------
    def save_trajectory_to_csv(self, trajectory, filename=None):
        """
        Save trajectory to CSV file with timestamp.
        Exports angles in degrees, rounded to 2 decimal places.
        
        Args:
            trajectory: list of numpy arrays (joint angles in radians)
            filename: optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"task_trajectory_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
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
        print("\n=== Task Space Trajectory Generator (Straight Line) ===")
        print("Note: This solver only controls position (X,Y,Z), not orientation.")
        print("The end-effector orientation depends on the joint configuration.\n")

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
            
            # Option to lock wrist angles
            lock_wrist = input("Lock wrist angles (q4, q5) to maintain orientation? (y/n): ").strip().lower()
            self.lock_wrist = lock_wrist == 'y'
            
            if self.lock_wrist:
                print("Wrist angles will be maintained from initial configuration")

            save_option = input("Save trajectory to CSV? (y/n): ").strip().lower()
            save_to_csv = save_option == 'y'

        except ValueError:
            print("Invalid input, try again.")
            return

        # Convert to IK units (millimeters)
        X0 = np.array([self.x0*1000, self.y0*1000, self.z0*1000])
        Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

        # Calculate trajectory distance
        distance = np.linalg.norm(Xf - X0)
        print(f"\nTrajectory distance: {distance:.2f} mm ({distance/1000:.4f} m)")

        # Solve IK for initial position using current robot state as initial guess
        print("Solving inverse kinematics for initial pose...")
        print(f"Using current joint angles as initial guess: {np.round(self.ik.q_current, 4)}")
        q0, conv0 = self.ik.inv_kinematics(self.ik.q_current.copy(), X0)
        if not conv0:
            print("IK failed for initial point.")
            return
        
        # Store wrist angles if locking orientation
        if self.lock_wrist:
            self.locked_q4 = q0[3]
            self.locked_q5 = q0[4]
            print(f"\n*** Locking wrist angles: q4={np.rad2deg(self.locked_q4):.2f}°, q5={np.rad2deg(self.locked_q5):.2f}° ***")
        
        print(f"Initial joint solution: {np.round(q0, 4)} rad")
        print(f"Initial joint solution: {np.round(np.rad2deg(q0), 2)}°")

        # Move to initial position (should be close to current position)
        msg = Float64MultiArray()
        msg.data = q0.tolist()
        self.joint_pub.publish(msg)
        self.get_logger().info(f"Adjusted to initial pose q0: {np.round(q0,4)}")
        self.get_logger().info(f"Initial pose (degrees): {np.round(np.rad2deg(q0),2)}")
        time.sleep(1.0)
        
        # Update IK node's current state
        self.ik.q_current = q0.copy()

        # Generate Cartesian trajectory waypoints
        print("Generating straight line trajectory in task space...")
        cartesian_traj = self.compute_cartesian_trajectory(X0, Xf, Tf, Ts)
        
        # Solve IK for each waypoint
        print(f"Solving IK for {len(cartesian_traj)} waypoints...")
        joint_traj = []
        q_current = q0.copy()
        
        for i, X_target in enumerate(cartesian_traj):
            q_solved, converged = self.ik.inv_kinematics(q_current.copy(), X_target)
            if not converged:
                print(f"Warning: IK failed at waypoint {i+1}/{len(cartesian_traj)}")
                print(f"Target position: {X_target}")
                # Use last successful solution
                joint_traj.append(q_current)
            else:
                # Lock wrist angles if requested
                if self.lock_wrist:
                    q_solved[3] = self.locked_q4
                    q_solved[4] = self.locked_q5
                
                joint_traj.append(q_solved)
                q_current = q_solved.copy()
            
            # Progress indicator
            if (i+1) % 10 == 0 or i == len(cartesian_traj)-1:
                print(f"  Progress: {(i+1)/len(cartesian_traj)*100:.1f}% ({i+1}/{len(cartesian_traj)})")

        print("IK solving complete.\n")

        # Save first trajectory if requested
        if save_to_csv:
            filename = f"task_trajectory_{traj_counter}.csv"
            self.save_trajectory_to_csv(joint_traj, filename)
            traj_counter += 1

        # Execute first trajectory
        print(f"\nExecuting trajectory with {len(joint_traj)} steps...")
        for i, q in enumerate(joint_traj):
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            self.get_logger().info(f"Step {i+1}/{len(joint_traj)}: {np.round(q,3)}")
            time.sleep(Ts)
        
        # Update IK node's current state
        self.ik.q_current = joint_traj[-1].copy()

        # -----------------------------------------
        # Subsequent movements
        # -----------------------------------------
        current_q = joint_traj[-1].copy()  # start next movement from last position
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

            # Current position from FK
            T_current = self.ik.fk_node.fwd_kinematics(current_q.tolist())
            X_current = T_current[:3, 3]
            
            # New final position
            Xf = np.array([self.xf*1000, self.yf*1000, self.zf*1000])

            # Calculate trajectory distance
            distance = np.linalg.norm(Xf - X_current)
            print(f"Trajectory distance: {distance:.2f} mm ({distance/1000:.4f} m)")

            # Generate Cartesian trajectory
            print("Generating trajectory...")
            cartesian_traj = self.compute_cartesian_trajectory(X_current, Xf, Tf, Ts)
            
            # Solve IK for each waypoint
            print(f"Solving IK for {len(cartesian_traj)} waypoints...")
            joint_traj = []
            
            for i, X_target in enumerate(cartesian_traj):
                q_solved, converged = self.ik.inv_kinematics(current_q.copy(), X_target)
                if not converged:
                    print(f"Warning: IK failed at waypoint {i+1}/{len(cartesian_traj)}")
                    joint_traj.append(current_q)
                else:
                    # Lock wrist angles if requested
                    if self.lock_wrist:
                        q_solved[3] = self.locked_q4
                        q_solved[4] = self.locked_q5
                    
                    joint_traj.append(q_solved)
                    current_q = q_solved.copy()
                
                # Progress indicator
                if (i+1) % 10 == 0 or i == len(cartesian_traj)-1:
                    print(f"  Progress: {(i+1)/len(cartesian_traj)*100:.1f}% ({i+1}/{len(cartesian_traj)})")

            print("IK solving complete.\n")

            # Save trajectory if requested
            if save_to_csv:
                filename = f"task_trajectory_{traj_counter}.csv"
                self.save_trajectory_to_csv(joint_traj, filename)
                traj_counter += 1

            # Execute trajectory
            print(f"\nExecuting trajectory with {len(joint_traj)} steps...")
            for i, q in enumerate(joint_traj):
                msg = Float64MultiArray()
                msg.data = q.tolist()
                self.joint_pub.publish(msg)
                self.get_logger().info(f"Step {i+1}/{len(joint_traj)}: {np.round(q,3)}")
                time.sleep(Ts)

            current_q = joint_traj[-1].copy()
            # Update IK node's current state
            self.ik.q_current = current_q.copy()

    # ------------------------------------------------------------
    # Modified IK that penalizes orientation changes
    # ------------------------------------------------------------
    def solve_ik_with_orientation_bias(self, q_init, X_target, orientation_weight=0.1):
        """
        Solve IK while trying to maintain similar joint configuration.
        This helps keep the end-effector orientation consistent.
        
        Args:
            q_init: initial joint angles
            X_target: target position [x, y, z] in mm
            orientation_weight: weight for penalizing joint changes
        """
        # First try standard IK
        q_solution, converged = self.ik.inv_kinematics(q_init.copy(), X_target)
        
        if not converged:
            return q_solution, False
        
        # If the solution changes joints drastically (especially q4 which affects wrist orientation),
        # try to find a better solution by biasing toward maintaining q4
        joint_change = np.abs(q_solution - q_init)
        
        # If q4 (wrist pitch) changed significantly, this might flip the end-effector
        if joint_change[3] > np.pi/4:  # More than 45 degrees change in q4
            self.get_logger().warn(f"Large q4 change detected: {np.rad2deg(joint_change[3]):.1f}°")
            self.get_logger().warn("End-effector orientation may have flipped")
        
        return q_solution, converged

    # ------------------------------------------------------------
    # Linear interpolation trajectory (straight line in Cartesian space)
    # ------------------------------------------------------------
    def compute_cartesian_trajectory(self, X0, Xf, Tf, Ts):
        """
        Compute straight line trajectory from X0 to Xf in Cartesian space.
        Uses linear interpolation for x, y, z coordinates.
        
        Args:
            X0: initial position [x, y, z] in millimeters
            Xf: final position [x, y, z] in millimeters
            Tf: total time (seconds)
            Ts: time step (seconds)
            
        Returns:
            List of position vectors along the straight line
        """
        steps = int(Tf / Ts)
        traj = []
        
        for k in range(steps + 1):
            # Linear interpolation parameter
            alpha = k / steps if steps > 0 else 1.0
            
            # Linear interpolation for each coordinate
            X_t = X0 + alpha * (Xf - X0)
            traj.append(X_t)
        
        return traj


def main(args=None):
    rclpy.init(args=args)
    node = TaskSpaceTrajectoryNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()