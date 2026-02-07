#!/usr/bin/env python3
"""
IK Position Controller Node

Subscribes to target poses, computes IK, sends trajectory commands
to the joint_trajectory_controller via ros2_control.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sim_utils_pkg.kinematics import IKSolver, Pose, IKResult


class IKPositionNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    def __init__(self):
        super().__init__('ik_position_node')

        # Parameters
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('trajectory_duration', 2.0)
        urdf_path = self.get_parameter('urdf_path').value
        self.trajectory_duration = self.get_parameter('trajectory_duration').value

        if not urdf_path:
            urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'

        # Initialize IK solver
        self.solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')
        self.get_logger().info(f'IK solver initialized: {self.solver.num_joints} joints')

        # Current joint state (seeded from joint_state_broadcaster)
        self.current_joints = [0.0] * 6

        # Publisher: trajectory command to the controller
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber: read current joints from joint_state_broadcaster
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        # Subscriber: target pose input
        self.create_subscription(PoseStamped, '/target_pose', self.target_pose_cb, 10)

        self.get_logger().info('Ready. Publish target pose to /target_pose')

    def joint_state_cb(self, msg: JointState):
        """Update current joint positions from joint_state_broadcaster."""
        for i, name in enumerate(self.JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joints[i] = msg.position[idx]

    def target_pose_cb(self, msg: PoseStamped):
        """Compute IK for target pose and send trajectory to controller."""
        p = msg.pose.position
        q = msg.pose.orientation
        target = Pose(
            position=(p.x, p.y, p.z),
            orientation=(q.x, q.y, q.z, q.w)
        )

        seed_deg = [f"{j*57.3:.1f}" for j in self.current_joints]
        self.get_logger().info(f'Target: pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        self.get_logger().info(f'IK seed (deg): {seed_deg}')

        # Solve IK using current joints as seed
        result, joints = self.solver.solve(target, self.current_joints)

        if result != IKResult.SUCCESS:
            self.get_logger().warn(f'IK failed: {result.name} (seed was {seed_deg})')
            return

        # Build trajectory message with a single waypoint
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joints.tolist()
        point.time_from_start = Duration(seconds=self.trajectory_duration).to_msg()

        traj_msg.points = [point]

        self.traj_pub.publish(traj_msg)
        self.get_logger().info(f'Sent trajectory (deg): {[f"{j*57.3:.1f}" for j in joints]}')


def main(args=None):
    rclpy.init(args=args)
    node = IKPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
