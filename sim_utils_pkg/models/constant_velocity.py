# models/constant_velocity.py

import numpy as np


class ConstantVelocityModel:
    """
    2D Constant Velocity Motion Model
    
    State:       x = [px, py, vx, vy]
    Measurement: z = [px, py]
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=0.5):
        # Dimensions
        self.dim_x = 4  # [px, py, vx, vy]
        self.dim_z = 2  # [px, py]
        self.dim_u = 0  # No control
        
        # Noise standard deviations
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def F(self, dt):
        """State transition matrix."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
    
    def H(self):
        """Measurement matrix."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    
    def Q(self, dt):
        """Process noise covariance."""
        q = self.process_noise ** 2
        return np.array([
            [q*dt**4/4, 0,         q*dt**3/2, 0        ],
            [0,         q*dt**4/4, 0,         q*dt**3/2],
            [q*dt**3/2, 0,         q*dt**2,   0        ],
            [0,         q*dt**3/2, 0,         q*dt**2  ]
        ])
    
    def R(self):
        """Measurement noise covariance."""
        r = self.measurement_noise ** 2
        return np.array([
            [r, 0],
            [0, r]
        ])
    
    def initial_state(self, position=(0, 0), velocity=(0, 0)):
        """Create initial state vector."""
        px, py = position
        vx, vy = velocity
        return np.array([px, py, vx, vy])
    
    def initial_covariance(self, pos_std=1.0, vel_std=1.0):
        """Create initial covariance matrix."""
        return np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2])