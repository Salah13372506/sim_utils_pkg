#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from turtlesim.msg import Pose
import math


class TurtlePIDController(Node):
    def __init__(self):
        super().__init__('turtle_pid_controller')
        
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.goal_sub = self.create_subscription(Pose2D, '/desired_pose', self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.current_pose = None
        self.goal_pose = None
        
        # PID gains
        self.kp_linear = 1.5
        self.kp_angular = 6.0
        
        self.distance_tolerance = 0.1
        self.angle_tolerance = 0.01
        
        self.get_logger().info('PID Controller initialized')
    
    def pose_callback(self, msg):
        self.current_pose = msg
    
    def goal_callback(self, msg):
        self.goal_pose = msg
        self.get_logger().info(f'New goal: x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}')
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def control_loop(self):
        if self.current_pose is None or self.goal_pose is None:
            return
        
        cmd = Twist()
        
        dx = self.goal_pose.x - self.current_pose.x
        dy = self.goal_pose.y - self.current_pose.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > self.distance_tolerance:
            angle_to_goal = math.atan2(dy, dx)
            angle_error = self.normalize_angle(angle_to_goal - self.current_pose.theta)
            
            cmd.linear.x = self.kp_linear * distance
            cmd.angular.z = self.kp_angular * angle_error
            
            cmd.linear.x = max(min(cmd.linear.x, 2.0), 0.0)
            cmd.angular.z = max(min(cmd.angular.z, 2.0), -2.0)
        else:
            angle_error = self.normalize_angle(self.goal_pose.theta - self.current_pose.theta)
            
            if abs(angle_error) > self.angle_tolerance:
                cmd.angular.z = self.kp_angular * angle_error
                cmd.angular.z = max(min(cmd.angular.z, 2.0), -2.0)
            else:
                self.get_logger().info('Goal reached')
                self.goal_pose = None
        
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = TurtlePIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()