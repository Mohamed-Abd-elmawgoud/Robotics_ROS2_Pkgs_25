#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

class JointStateToTrajectory(rclpy.node.Node):
    def __init__(self):
        super().__init__('joint_state_to_trajectory')
        
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, 
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Listen to slider commands, not actual robot state
        self.subscription = self.create_subscription(
            JointState, '/joint_state_commands', self.joint_state_callback, 10
        )
        
        self.last_positions = None
        self.get_logger().info('Joint State to Trajectory converter started')

    def joint_state_callback(self, msg):
        # Updated joint names for 5-DOF robot
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5']
        
        positions = []
        for joint_name in joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
            else:
                positions.append(0.0)
        
        # Only send if positions changed
        if self.last_positions is None or positions != self.last_positions:
            self.last_positions = positions
            
            traj = JointTrajectory()
            traj.header.frame_id = 'world'
            traj.joint_names = joint_names
            
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0]
            point.time_from_start = Duration(sec=1, nanosec=0)
            
            traj.points.append(point)
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            
            self.trajectory_client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    node = JointStateToTrajectory()
    rclpy.spin(node)

if __name__ == '__main__':
    main()