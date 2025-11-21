# sim_utils_pkg/nodes/kalman_test_node.py

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray

from sim_utils_pkg.estimators.kalman_filter import KalmanFilter
from sim_utils_pkg.models.constant_velocity import ConstantVelocityModel


class KalmanTestNode(Node):
    """
    Test node for Kalman Filter validation.
    
    Simulates a 2D trajectory, adds measurement noise,
    filters with Kalman, and publishes results for visualization.
    
    Published topics:
        /ground_truth   (PoseStamped)  - True position
        /measurement    (PoseStamped)  - Noisy measurement
        /estimated      (PoseStamped)  - Kalman estimate
        /uncertainty    (Marker)       - Covariance ellipse
        /trajectory     (PoseArray)    - Full trajectory history
    """
    
    def __init__(self):
        super().__init__('kalman_test_node')
        
        # Parameters
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('process_noise', 0.1)
        self.declare_parameter('measurement_noise', 0.5)
        self.declare_parameter('trajectory_type', 'circle')  # 'circle' or 'line'
        
        self.dt = self.get_parameter('dt').value
        process_noise = self.get_parameter('process_noise').value
        measurement_noise = self.get_parameter('measurement_noise').value
        self.trajectory_type = self.get_parameter('trajectory_type').value
        
        # Initialize model and filter
        self.model = ConstantVelocityModel(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        self.kf = KalmanFilter(
            dim_x=self.model.dim_x,
            dim_u=self.model.dim_u,
            dim_z=self.model.dim_z
        )
        
        # Initial state
        x0 = self.model.initial_state(position=(0, 0), velocity=(1, 0))
        P0 = self.model.initial_covariance(pos_std=0.5, vel_std=1.0)
        self.kf.set_state(x0, P0)
        
        # Simulation state
        self.time = 0.0
        self.true_state = np.array([0.0, 0.0, 1.0, 0.0])  # [px, py, vx, vy]
        
        # History for trajectory visualization
        self.history_true = []
        self.history_measured = []
        self.history_estimated = []
        
        # Publishers
        self.pub_true = self.create_publisher(PoseStamped, 'ground_truth', 10)
        self.pub_meas = self.create_publisher(PoseStamped, 'measurement', 10)
        self.pub_est = self.create_publisher(PoseStamped, 'estimated', 10)
        self.pub_ellipse = self.create_publisher(Marker, 'uncertainty', 10)
        self.pub_traj = self.create_publisher(PoseArray, 'trajectory', 10)
        self.pub_error = self.create_publisher(Float64MultiArray, 'error', 10)
        
        # Timer
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(
            f'Kalman test started: trajectory={self.trajectory_type}, '
            f'dt={self.dt}, Q={process_noise}, R={measurement_noise}'
        )
    
    def timer_callback(self):
        """Main loop: simulate, filter, publish."""
        
        # 1. Update true state (ground truth)
        self.true_state = self._simulate_true_state()
        true_pos = self.true_state[:2]
        
        # 2. Generate noisy measurement
        noise = np.random.randn(2) * self.model.measurement_noise
        measurement = true_pos + noise
        
        # 3. Kalman predict + update
        F = self.model.F(self.dt)
        Q = self.model.Q(self.dt)
        H = self.model.H()
        R = self.model.R()
        
        self.kf.predict(F, Q)
        self.kf.update(measurement, H, R)
        
        estimated, covariance = self.kf.get_state()
        estimated_pos = estimated[:2]
        
        # 4. Store history
        self.history_true.append(true_pos.copy())
        self.history_measured.append(measurement.copy())
        self.history_estimated.append(estimated_pos.copy())
        
        # Keep last 500 points
        max_history = 500
        if len(self.history_true) > max_history:
            self.history_true.pop(0)
            self.history_measured.pop(0)
            self.history_estimated.pop(0)
        
        # 5. Publish
        stamp = self.get_clock().now().to_msg()
        
        self._publish_pose(self.pub_true, true_pos, stamp)
        self._publish_pose(self.pub_meas, measurement, stamp)
        self._publish_pose(self.pub_est, estimated_pos, stamp)
        self._publish_ellipse(estimated_pos, covariance, stamp)
        self._publish_error(true_pos, estimated_pos)
        
        # Log every 2 seconds
        self.time += self.dt
        if int(self.time / self.dt) % 20 == 0:
            error = np.linalg.norm(true_pos - estimated_pos)
            sigma = self.kf.get_uncertainty()[:2]
            self.get_logger().info(
                f't={self.time:.1f}s | error={error:.3f}m | σ={sigma}'
            )
    
    def _simulate_true_state(self):
        """Generate ground truth trajectory."""
        if self.trajectory_type == 'circle':
            # Circular motion: radius=5m, angular velocity=0.2 rad/s
            omega = 0.2
            radius = 5.0
            angle = omega * self.time
            
            px = radius * np.cos(angle)
            py = radius * np.sin(angle)
            vx = -radius * omega * np.sin(angle)
            vy = radius * omega * np.cos(angle)
            
        else:  # 'line'
            # Linear motion with slight acceleration
            px = self.time * 1.0
            py = self.time * 0.5 + 0.1 * np.sin(self.time)
            vx = 1.0
            vy = 0.5 + 0.1 * np.cos(self.time)
        
        return np.array([px, py, vx, vy])
    
    def _publish_pose(self, publisher, position, stamp):
        """Publish position as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = 0.0
        publisher.publish(msg)
    
    def _publish_ellipse(self, position, covariance, stamp):
        """Publish covariance as ellipse marker."""
        msg = Marker()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.id = 0
        
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = 0.0
        
        # 2-sigma ellipse (95% confidence)
        sigma = np.sqrt(np.diag(covariance))
        msg.scale.x = float(4 * sigma[0])  # 2-sigma diameter
        msg.scale.y = float(4 * sigma[1])
        msg.scale.z = 0.1
        
        msg.color.r = 0.0
        msg.color.g = 0.5
        msg.color.b = 1.0
        msg.color.a = 0.3
        
        self.pub_ellipse.publish(msg)
    
    def _publish_error(self, true_pos, estimated_pos):
        """Publish estimation error."""
        error = np.linalg.norm(true_pos - estimated_pos)
        msg = Float64MultiArray()
        msg.data = [error, float(true_pos[0]), float(true_pos[1]),
                    float(estimated_pos[0]), float(estimated_pos[1])]
        self.pub_error.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KalmanTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()