#!/usr/bin/env python3
"""
IK Position Controller Node

Subscribes to target poses, computes IK, publishes joint states.
For use with RViz visualization (no controller needed).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

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
        urdf_path = self.get_parameter('urdf_path').value

        if not urdf_path:
            # Default path
            urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'

        # Initialize IK solver
        self.solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')
        self.get_logger().info(f'IK solver initialized: {self.solver.num_joints} joints')

        # Current joint state
        self.current_joints = [0.0] * 6

        # Publishers & Subscribers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/target_pose', self.target_pose_cb, 10)

        self.get_logger().info('Ready. Publish target pose to /target_pose')

    def joint_state_cb(self, msg: JointState):
        """Store current joint positions."""
        updated = False
        for i, name in enumerate(self.JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                if abs(self.current_joints[i] - msg.position[idx]) > 0.01:
                    updated = True
                self.current_joints[i] = msg.position[idx]

        # Log current pose when joints change significantly
        if updated:
            pose = self.solver.forward(self.current_joints)
            if pose:
                p = pose.position
                o = pose.orientation
                self.get_logger().info(
                    f'Current EE: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) '
                    f'quat=({o[3]:.3f}, {o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f})'  # w,x,y,z
                )

    def target_pose_cb(self, msg: PoseStamped):
        """Compute IK for target pose and publish result."""
        # Convert ROS pose to our Pose type
        p = msg.pose.position
        q = msg.pose.orientation
        target = Pose(
            position=(p.x, p.y, p.z),
            orientation=(q.x, q.y, q.z, q.w)
        )

        self.get_logger().info(f'Target: pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

        # Solve IK
        result, joints = self.solver.solve(target, self.current_joints)

        if result != IKResult.SUCCESS:
            self.get_logger().warn(f'IK failed: {result.name}')
            return

        # Publish joint state
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.JOINT_NAMES
        joint_msg.position = joints.tolist()

        self.joint_pub.publish(joint_msg)
        self.current_joints = joints.tolist()

        self.get_logger().info(f'Published joints (deg): {[f"{j*57.3:.1f}" for j in joints]}')


def main(args=None):
    rclpy.init(args=args)
    node = IKPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
