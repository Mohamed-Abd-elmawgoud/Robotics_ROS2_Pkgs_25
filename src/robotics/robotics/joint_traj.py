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
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_angles_out', 10)

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
        q_sol = self.ik.solve_ik_full(q_init, ee_pos_mm, q5_rad)
        if q_sol is None:
            self.get_logger().error("❌ IK FAILED - No solution found")
            return None, False

        # Verify solution
        # Flatten joint solution
        if isinstance(q_sol, tuple):
            q_flat = np.hstack([np.atleast_1d(x) for x in q_sol])
        else:
            q_flat = np.array(q_sol)
        
        # Ensure we only have 5 joints
        q_flat = q_flat[:5]

        T = self.ik.fk_node.fwd_kinematics(q_flat.tolist())
        pos_error = np.linalg.norm(T[:3, 3] - ee_pos_mm)
        R33 = self.ik.get_R33(q_flat)  # Use q_flat instead of q_sol
        if pos_error > 10.0 or R33 < 0.85:
            self.get_logger().warn(
                f"⚠ Accepting IK solution with errors: pos_err={pos_error:.3f} mm, R33={R33:.4f}"
            )

        return q_flat, True  # Return q_flat instead of q_sol

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

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def run_loop(self):
        print("\n" + "="*70)
        print("CONTINUOUS POSE TRAJECTORY GENERATOR")
        print("Tool Always Pointing DOWN")
        print("="*70)
        print("\nEnter positions in meters and q5 rotation in degrees")
        print("Type 'quit' or 'q' to exit\n")
        
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
                    
                    X0 = xyz0 * 1000
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

            # Solve IK for initial and final
            if first_trajectory:
                q_guess = np.zeros(5)
                q0, conv0 = self.solve_ik_for_pose(X0, q5_0_rad, q_guess)
                 # Set q4 to 0 for initial configuration
                q0[3] = 0
                if not conv0:
                    print("❌ IK FAILED FOR INITIAL POSITION")
                    print("Please try different values.")
                    continue
            else:
                # Use previous final joints as initial
                q0 = self.current_q.copy()
               

                
            
            # Solve IK for final position using initial joints as guess
            qf, convf = self.solve_ik_for_pose(Xf, q5_f_rad, q0.copy())

            if Xf[0] == 0 and Xf[1]==0 and Xf[2] == -403 :
                qf[3] = 0
            
            if not convf:
                print("❌ IK FAILED FOR FINAL POSITION")
                print("Please try different values.")
                continue

            print(f"\n✓ Initial joints: {np.round(np.rad2deg(q0), 1)}°")
            print(f"✓ Final joints:   {np.round(np.rad2deg(qf), 1)}°")

           

            # Generate trajectory
            trajectory = self.compute_trajectory(q0, qf, self.Tf, self.Ts)
            if save_to_csv:
                self.save_trajectory_to_csv(trajectory, f"trajectory_{traj_counter}.csv")

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
            self.current_q = qf.copy()
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


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrajectoryNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()