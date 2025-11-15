#!/usr/bin/env python3

import numpy as np

# FK class
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
            theta = [
                -np.pi/2 + q[0],
                -np.pi/2 + q[1],
                q[2],
                -np.pi/2 + q[3],
                np.pi/2 + q[4]
            ]
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


class VelocityKinematicsInteractive:
    def __init__(self):
        self.fk_node = ForwardKinematicsNode()
        self.q = np.zeros(5)
        self.q_dot = np.zeros(5)
        self.X_dot_desired = np.zeros(3)

        # Ask mode
        print("\nSelect velocity mode:")
        print("1: Forward velocity (joint velocities → Cartesian velocity)")
        print("2: Inverse velocity (Cartesian velocity → joint velocities)")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            self.mode = "forward"
            print("Forward velocity mode selected.")
        elif choice == "2":
            self.mode = "inverse"
            print("Inverse velocity mode selected.")
        else:
            print("Invalid mode. Exiting.")
            exit()

        self.run()

    # ----------------------------------------
    # Jacobian and pseudo-inverse
    # ----------------------------------------
    def jacobian_matrix(self, q):
        epsilon = 1e-6
        J = np.zeros((3, 5))
        pos_current = self.fk_node.fwd_kinematics(q.tolist())[:3, 3]
        for i in range(5):
            q_pert = q.copy()
            q_pert[i] += epsilon
            pos_pert = self.fk_node.fwd_kinematics(q_pert.tolist())[:3, 3]
            J[:, i] = (pos_pert - pos_current) / epsilon
        return J

    def inverse_jacobian_matrix(self, q):
        J = self.jacobian_matrix(q)
        JJT = J @ J.T
        if abs(np.linalg.det(JJT)) < 1e-6:
            q[1] += 0.1
            q[2] += 0.1
            J = self.jacobian_matrix(q)
            JJT = J @ J.T
        try:
            return J.T @ np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(J)

    # ----------------------------------------
    # Forward velocity
    # ----------------------------------------
    def fwd_velocity_kinematics(self, q, q_dot):
        return self.jacobian_matrix(q) @ q_dot

    # ----------------------------------------
    # Inverse velocity
    # ----------------------------------------
    def inv_velocity_kinematics(self, q, X_dot):
        return self.inverse_jacobian_matrix(q) @ X_dot

    # ----------------------------------------
    # Interactive loop
    # ----------------------------------------
    def run(self):
        # Get joint angles in degrees
        q_input = input("Enter 5 joint angles in degrees, separated by spaces: ")
        q_deg = np.array([float(x) for x in q_input.strip().split()])
        if len(q_deg) != 5:
            print("You must enter exactly 5 joint angles. Exiting.")
            exit()
        # Convert degrees to radians
        self.q = np.radians(q_deg)

        if self.mode == "forward":
            q_dot_input = input("Enter 5 joint velocities (rad/s), separated by spaces: ")
            self.q_dot = np.array([float(x) for x in q_dot_input.strip().split()])
            if len(self.q_dot) != 5:
                print("You must enter exactly 5 joint velocities. Exiting.")
                exit()
            X_dot = self.fwd_velocity_kinematics(self.q, self.q_dot)
            print("\nComputed Cartesian velocity [vx, vy, vz] (mm/s):")
            print(np.round(X_dot, 4))
        else:  # inverse
            X_dot_input = input("Enter desired Cartesian velocity [vx, vy, vz] in mm/s, separated by spaces: ")
            self.X_dot_desired = np.array([float(x) for x in X_dot_input.strip().split()])
            if len(self.X_dot_desired) != 3:
                print("You must enter exactly 3 Cartesian velocities. Exiting.")
                exit()
            q_dot = self.inv_velocity_kinematics(self.q, self.X_dot_desired)
            print("\nComputed joint velocities [q1_dot, ..., q5_dot] (rad/s):")
            print(np.round(q_dot, 4))


def main():
    VelocityKinematicsInteractive()


if __name__ == "__main__":
    main()
