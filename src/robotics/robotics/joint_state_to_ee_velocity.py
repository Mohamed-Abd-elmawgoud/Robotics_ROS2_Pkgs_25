#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R


class JointStateToEEVelocity(Node):
    def __init__(self):
        super().__init__('joint_state_to_ee_velocity')
        
        # REMOVED: self.declare_parameter('use_sim_time', True)
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.vel_pub = self.create_publisher(TwistStamped, '/ee_velocity', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        
        self.get_logger().info('Computing EE velocity from joint states using Jacobian')
    
    def joint_callback(self, msg):
        try:
            # Extract joint positions and velocities
            joint_indices = {name: idx for idx, name in enumerate(msg.name)}
            
            required_joints = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5']
            if not all(joint in joint_indices for joint in required_joints):
                return
            
            q = np.array([
                msg.position[joint_indices['Joint_1']],
                msg.position[joint_indices['Joint_2']],
                msg.position[joint_indices['Joint_3']],
                msg.position[joint_indices['Joint_4']],
                msg.position[joint_indices['Joint_5']]
            ])
            
            q_dot = np.array([
                msg.velocity[joint_indices['Joint_1']],
                msg.velocity[joint_indices['Joint_2']],
                msg.velocity[joint_indices['Joint_3']],
                msg.velocity[joint_indices['Joint_4']],
                msg.velocity[joint_indices['Joint_5']]
            ])
            
            # Compute forward kinematics
            ee_pos, ee_rot = self.forward_kinematics(q)
            
            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose.position.x = float(ee_pos[0])
            pose_msg.pose.position.y = float(ee_pos[1])
            pose_msg.pose.position.z = float(ee_pos[2])
            quat = ee_rot.as_quat()
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            self.pose_pub.publish(pose_msg)
            
            # Compute Jacobian
            J = self.compute_jacobian(q)
            
            # Compute EE velocity
            ee_twist = J @ q_dot
            
            # Publish velocity
            vel_msg = TwistStamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.header.frame_id = 'base_link'
            vel_msg.twist.linear.x = float(ee_twist[0])
            vel_msg.twist.linear.y = float(ee_twist[1])
            vel_msg.twist.linear.z = float(ee_twist[2])
            vel_msg.twist.angular.x = float(ee_twist[3])
            vel_msg.twist.angular.y = float(ee_twist[4])
            vel_msg.twist.angular.z = float(ee_twist[5])
            
            self.vel_pub.publish(vel_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}', throttle_duration_sec=5.0)
    
    def forward_kinematics(self, q):
        """Forward kinematics based on URDF"""
        T = np.eye(4)
        
        # Joint 1
        T1 = np.eye(4)
        T1[:3, 3] = [0.042627, -0.088458, 0.04544]
        T1[:3, :3] = R.from_euler('xyz', [0, -np.pi/2, np.pi]).as_matrix()
        R1 = R.from_euler('y', q[0]).as_matrix()
        T1[:3, :3] = T1[:3, :3] @ R1
        T = T @ T1
        
        # Joint 2
        T2 = np.eye(4)
        T2[:3, 3] = [0, 0.04355, 0]
        T2[:3, :3] = R.from_euler('xyz', [np.pi/2, 0, 1.6019]).as_matrix()
        R2 = R.from_euler('y', q[1]).as_matrix()
        T2[:3, :3] = T2[:3, :3] @ R2
        T = T @ T2
        
        # Joint 3
        T3 = np.eye(4)
        T3[:3, 3] = [0.14, 0, 0]
        T3[:3, :3] = R.from_euler('xyz', [np.pi/2, -0.031075, 0]).as_matrix()
        R3 = R.from_euler('z', q[2]).as_matrix()
        T3[:3, :3] = T3[:3, :3] @ R3
        T = T @ T3
        
        # Joint 4
        T4 = np.eye(4)
        T4[:3, 3] = [0.1329, 0.0005, 0]
        R4 = R.from_euler('z', q[3]).as_matrix()
        T4[:3, :3] = R4
        T = T @ T4
        
        # Joint 5
        T5 = np.eye(4)
        T5[:3, 3] = [0.02245, -0.00038473, -0.0018312]
        T5[:3, :3] = R.from_euler('xyz', [np.pi/2, 0, 0]).as_matrix()
        R5 = R.from_euler('x', q[4]).as_matrix()
        T5[:3, :3] = T5[:3, :3] @ R5
        T = T @ T5
        
        # EE frame
        T_ee = np.eye(4)
        T_ee[:3, 3] = [0.09, 0, 0]
        T = T @ T_ee
        
        position = T[:3, 3]
        rotation = R.from_matrix(T[:3, :3])
        
        return position, rotation
    
    def compute_jacobian(self, q):
        """Numerical Jacobian"""
        J = np.zeros((6, 5))
        epsilon = 1e-6
        
        ee_pos_0, ee_rot_0 = self.forward_kinematics(q)
        
        for i in range(5):
            q_p = q.copy()
            q_p[i] += epsilon
            
            ee_pos_p, ee_rot_p = self.forward_kinematics(q_p)
            
            # Linear velocity
            J[:3, i] = (ee_pos_p - ee_pos_0) / epsilon
            
            # Angular velocity
            rot_diff = ee_rot_0.inv() * ee_rot_p
            rotvec = rot_diff.as_rotvec()
            J[3:, i] = rotvec / epsilon
        
        return J


def main(args=None):
    rclpy.init(args=args)
    node = JointStateToEEVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()