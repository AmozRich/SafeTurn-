import numpy as np

class KalmanFilter:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, initial_state=0.0):
        """
        Initialize a 1D Kalman Filter for tracking position and velocity.
        
        State vector x = [position, velocity]
        """
        # 1. State Vector [pos, vel]
        self.x = np.array([[initial_state], [0.0]])
        
        # 2. State Transition Matrix F
        # pos = pos + vel*dt
        # vel = vel
        # (dt is assumed 1.0 per frame for simplicity in video processing, 
        #  or can be passed in predict)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        
        # 3. Measurement Matrix H
        # We only measure position
        self.H = np.array([[1.0, 0.0]])
        
        # 4. Covariance Matrix P (Initial uncertainty)
        self.P = np.array([[1000.0, 0.0],
                           [0.0, 1000.0]])
        
        # 5. Process Noise Q (Uncertainty in the model)
        # Variance of position and velocity
        self.Q = np.array([[process_noise, 0.0],
                           [0.0, process_noise]])
        
        # 6. Measurement Noise R (Uncertainty in the measurement)
        self.R = np.array([[measurement_noise]])

    def predict(self):
        """
        Predict the next state based on the previous state and physics model.
        x = F * x
        P = F * P * F.T + Q
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0, 0]

    def update(self, measurement):
        """
        Update the state with a new measurement calculation.
        y = z - H * x (Residual)
        S = H * P * H.T + R (Residual Covariance)
        K = P * H.T * S^-1 (Kalman Gain)
        x = x + K * y
        P = (I - K * H) * P
        """
        # Measurement vector z
        z = np.array([[measurement]])
        
        # Error (Residual)
        y = z - np.dot(self.H, self.x)
        
        # System uncertainty maps to measurement space
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain
        # S is 1x1 scalar, so divide avoiding matrix inversion
        K = np.dot(self.P, self.H.T) / S[0, 0]
        
        # Update State
        self.x = self.x + np.dot(K, y)
        
        # Update Covariance
        I = np.eye(self.F.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x[0, 0]

    def get_position(self):
        return self.x[0, 0]
        
    def get_velocity(self):
        return self.x[1, 0]
