# sim_utils_pkg/estimators/kalman_filter.py

import numpy as np


class KalmanFilter:
    """
    Generic Linear Kalman Filter for any linear system.
    
    System model:
        x(k+1) = F*x(k) + B*u(k) + w    where w ~ N(0, Q)
        z(k)   = H*x(k) + v              where v ~ N(0, R)
    
    The filter does NOT store system matrices (F, B, H, Q, R).
    They must be provided at each predict/update step for maximum flexibility.
    """
    
    def __init__(self, dim_x, dim_u, dim_z):
        """
        Initialize Kalman Filter with dimensions.
        
        Args:
            dim_x: Dimension of state vector
            dim_u: Dimension of control input (can be 0)
            dim_z: Dimension of measurement vector
        """
        if dim_x <= 0 or dim_u < 0 or dim_z <= 0:
            raise ValueError("Dimensions must be positive (dim_u can be 0)")
        
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_z = dim_z
        
        # State estimate and covariance
        self._x = np.zeros(dim_x)
        self._P = np.eye(dim_x)
        
        # Innovation and Kalman gain (for diagnostics)
        self._y = np.zeros(dim_z)
        self._S = np.eye(dim_z)
        self._K = np.zeros((dim_x, dim_z))
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def predict(self, F, Q, B=None, u=None):
        """
        Prediction step.
        
        x_prior = F @ x + B @ u
        P_prior = F @ P @ F.T + Q
        """
        self._validate_matrix(F, (self.dim_x, self.dim_x), "F")
        self._validate_matrix(Q, (self.dim_x, self.dim_x), "Q")
        
        # Predict state
        self._x = F @ self._x
        
        # Add control input if provided
        if B is not None and u is not None:
            self._validate_matrix(B, (self.dim_x, self.dim_u), "B")
            self._validate_vector(u, self.dim_u, "u")
            self._x += B @ u
        
        # Predict covariance
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)  # Ensure symmetry
    
    def update(self, z, H, R):
        """
        Measurement update step.
        
        y = z - H @ x              (innovation)
        S = H @ P @ H.T + R        (innovation covariance)
        K = P @ H.T @ inv(S)       (Kalman gain)
        x = x + K @ y              (updated state)
        P = (I - K @ H) @ P        (updated covariance)
        """
        self._validate_vector(z, self.dim_z, "z")
        self._validate_matrix(H, (self.dim_z, self.dim_x), "H")
        self._validate_matrix(R, (self.dim_z, self.dim_z), "R")
        
        # Innovation
        self._y = z - H @ self._x
        self._S = H @ self._P @ H.T + R
        
        # Kalman gain
        try:
            self._K = self._P @ H.T @ np.linalg.inv(self._S)
        except np.linalg.LinAlgError:
            self._K = self._P @ H.T @ np.linalg.pinv(self._S)
        
        # Update state
        self._x = self._x + self._K @ self._y
        
        # Update covariance (Joseph form)
        I_KH = np.eye(self.dim_x) - self._K @ H
        self._P = I_KH @ self._P @ I_KH.T + self._K @ R @ self._K.T
        self._P = 0.5 * (self._P + self._P.T)  # Ensure symmetry
    
    # =========================================================================
    # STATE ACCESS
    # =========================================================================
    
    @property
    def x(self):
        """Current state estimate."""
        return self._x.copy()
    
    @property
    def P(self):
        """Current covariance matrix."""
        return self._P.copy()
    
    @property
    def innovation(self):
        """Last innovation (residual)."""
        return self._y.copy()
    
    @property
    def kalman_gain(self):
        """Last Kalman gain."""
        return self._K.copy()
    
    def get_state(self):
        """Get (x, P) tuple."""
        return self.x, self.P
    
    def set_state(self, x, P):
        """Set state estimate and covariance."""
        self._validate_vector(x, self.dim_x, "x")
        self._validate_matrix(P, (self.dim_x, self.dim_x), "P")
        
        self._x = np.asarray(x).copy()
        self._P = np.asarray(P).copy()
        self._P = 0.5 * (self._P + self._P.T)
    
    def reset(self):
        """Reset to zero state, identity covariance."""
        self._x = np.zeros(self.dim_x)
        self._P = np.eye(self.dim_x)
        self._y = np.zeros(self.dim_z)
        self._S = np.eye(self.dim_z)
        self._K = np.zeros((self.dim_x, self.dim_z))
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def predict_measurement(self, H):
        """Predict measurement: z_pred = H @ x"""
        self._validate_matrix(H, (self.dim_z, self.dim_x), "H")
        return H @ self._x
    
    def mahalanobis_distance(self, z, H, R):
        """Mahalanobis distance for outlier detection."""
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        
        try:
            return np.sqrt(y.T @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            return np.inf
    
    def get_uncertainty(self):
        """Standard deviation for each state variable."""
        return np.sqrt(np.diag(self._P))
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_vector(self, vec, expected_dim, name):
        vec = np.asarray(vec)
        if vec.shape != (expected_dim,):
            raise ValueError(f"{name} must have shape ({expected_dim},), got {vec.shape}")
    
    def _validate_matrix(self, mat, expected_shape, name):
        mat = np.asarray(mat)
        if mat.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {mat.shape}")
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self):
        return f"KalmanFilter(dim_x={self.dim_x}, dim_u={self.dim_u}, dim_z={self.dim_z})"
    
    def __str__(self):
        sigma = self.get_uncertainty()
        return (f"KalmanFilter\n"
                f"  Dimensions: x={self.dim_x}, u={self.dim_u}, z={self.dim_z}\n"
                f"  State: {self._x}\n"
                f"  Uncertainty (σ): {sigma}")