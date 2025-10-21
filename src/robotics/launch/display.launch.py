from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_path = get_package_share_directory('robotics')
    pkg_share_parent = os.path.dirname(pkg_path)

    # Consistent sim time parameter for all nodes
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    urdf_file = os.path.join(pkg_path, 'urdf', 'robot.urdf')
    robot_controllers = os.path.join(pkg_path, 'config', 'controllers.yaml')
    robot_description = {'robot_description': open(urdf_file).read()}

    set_gz_env = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=f"{pkg_share_parent}:{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}"
    )

    # --- Core Nodes ---
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'robotics', '-allow_renaming', 'true'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', robot_controllers
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', robot_controllers
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # --- Custom Robotics Nodes ---
    tf_to_pose_node = Node(
        package='robotics',
        executable='tf_to_pose_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    joint_trajectory_node = Node(
        package='robotics',
        executable='joint_state_to_trajectory',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    fwd_kinematics_node = Node(
        package='robotics',
        executable='fwd_kinematics',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # --- Launch Description ---
    return LaunchDescription([
        set_gz_env,
        tf_to_pose_node,
        joint_trajectory_node,
        fwd_kinematics_node,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])]
            ),
            launch_arguments=[('gz_args', [' -r -v 4 empty.sdf'])],
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[joint_trajectory_controller_spawner],
            )
        ),
        bridge,
        node_robot_state_publisher,
        gz_spawn_entity,
        DeclareLaunchArgument('use_sim_time', default_value='true'),
    ])
