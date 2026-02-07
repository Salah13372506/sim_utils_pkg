"""
Launch ros2_control with JointTrajectoryController + IK position node.

Uses mock hardware (no Gazebo needed) for testing.
"""

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # --- Generate URDF with mock hardware enabled ---
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('ur_description'), 'urdf', 'ur.urdf.xacro'
        ]),
        ' ', 'name:=ur',
        ' ', 'ur_type:=ur5e',
        ' ', 'use_fake_hardware:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(
            value=robot_description_content, value_type=str
        )
    }

    # --- Controller config ---
    controllers_config = PathJoinSubstitution([
        FindPackageShare('sim_utils_pkg'), 'config', 'ros2_controllers.yaml'
    ])

    # --- Nodes ---

    # Controller manager: loads hardware interface + runs control loop
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, controllers_config],
        output='both',
    )

    # Spawn joint_state_broadcaster (publishes /joint_states)
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    # Spawn joint_trajectory_controller
    jtc_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        output='screen',
    )

    # Robot state publisher: URDF + /joint_states -> /tf
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='both',
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('ur_description'), 'rviz', 'view_robot.rviz'
        ])],
        output='log',
    )

    # IK position node
    ik_node = Node(
        package='sim_utils_pkg',
        executable='ik_position_node',
        parameters=[{
            'trajectory_duration': 2.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        control_node,
        robot_state_publisher,
        jsb_spawner,
        jtc_spawner,
        rviz,
        ik_node,
    ])
