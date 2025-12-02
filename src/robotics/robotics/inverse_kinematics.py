
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import numpy as np

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

        # FK node instance (used for numeric FK calculations)
        self.fk_node = ForwardKinematicsNode()

        # IK parameters
        self.error_tolerance = 0.001  # mm
        self.max_iterations = 10000

        # Joint limits (radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # q1
            [-np.pi/2, np.pi/2],  # q2
            [-np.pi/2, np.pi/2],  # q3
            [-np.pi/2, np.pi/2],  # q4
            [-np.pi, np.pi]       # q5
        ])

        # Current joint angles (initial guess)
        self.q_current = np.array([0.0, 0.2, 0.2, 0.2, 0.0])

        # Subscribers and publishers
        self.position_sub = self.create_subscription(
            Point, 'desired_position', self.position_callback, 10
        )

        self.joint_angles_pub = self.create_publisher(
            Float64MultiArray, 'joint_angles_out', 10
        )

        self.get_logger().info("Inverse Kinematics Node started with joint limits enforced.")

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
    # Compute numerical Jacobian (position only)
    # ------------------------------------------------------------
    def jacobian_matrix(self, q):
        epsilon = 1e-6
        J = np.zeros((3, 5))
        pos_current = self.fk_node.fwd_kinematics(q.tolist())[:3, 3]
        for i in range(5):
            q_perturbed = q.copy()
            q_perturbed[i] += epsilon
            pos_perturbed = self.fk_node.fwd_kinematics(q_perturbed.tolist())[:3, 3]
            J[:, i] = (pos_perturbed - pos_current) / epsilon
        return J

    # ------------------------------------------------------------
    # Compute inverse Jacobian (pseudo-inverse)
    # ------------------------------------------------------------
    def inverse_jacobian_matrix(self, q):
        J = self.jacobian_matrix(q)
        try:
            J_inv = J.T @ np.linalg.inv(J @ J.T)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular Jacobian, using np.linalg.pinv fallback")
            J_inv = np.linalg.pinv(J)
        return J_inv

    # ------------------------------------------------------------
    # Newton-Raphson IK solver
    # ------------------------------------------------------------
    def inv_kinematics(self, q0, X_desired):
        q_current = q0.copy()
        for iteration in range(self.max_iterations):
            pos_current = self.fk_node.fwd_kinematics(q_current.tolist())[:3, 3]
            F = pos_current - X_desired
            if np.linalg.norm(F) < self.error_tolerance:
                return q_current, True
            J_inv = self.inverse_jacobian_matrix(q_current)
            q_current += -J_inv @ F
            q_current = self.apply_joint_limits(q_current)
        return q_current, False

    # ------------------------------------------------------------
    # Callback for desired EE position
    # ------------------------------------------------------------
    def position_callback(self, msg):
        X_desired = np.array([msg.x*1000.0, msg.y*1000.0, msg.z*1000.0])
        self.get_logger().info(f"Desired position: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}] m")

        # Solve IK
        q_solution, converged = self.inv_kinematics(self.q_current, X_desired)
        if not converged:
            self.get_logger().error("IK did not converge - not publishing")
            return

        # Update current joint angles
        self.q_current = q_solution

        # Publish joint angles (radians) for forward position controller
        joint_msg = Float64MultiArray()
        joint_msg.data = q_solution.tolist()
        self.joint_angles_pub.publish(joint_msg)

        # Log solution
        self.get_logger().info(f"Joint angles (rad): {np.round(q_solution,4)}")
        self.get_logger().info(f"Joint angles (deg): {np.round(np.rad2deg(q_solution),2)}")

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
