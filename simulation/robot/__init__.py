"""
Low-level robot controller.

Implements control loops and command filtering.
"""

import numpy as np
from typing import Tuple


class RobotController:
    """PID-based motion controller for robot."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.1):
        """
        Args:
            kp, ki, kd: PID gains
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.dt = 0.1  # control timestep
    
    def compute_control(self, error: float) -> float:
        """
        Compute control signal from error using PID.
        
        Args:
            error: current tracking error
            
        Returns:
            control signal
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral_error += error * self.dt
        i_term = self.ki * self.integral_error
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        control = p_term + i_term + d_term
        return control
    
    def reset(self):
        """Reset controller state."""
        self.integral_error = 0.0
        self.prev_error = 0.0
    
    def saturate(self, value: float, min_val: float, max_val: float) -> float:
        """Saturate value to bounds."""
        return np.clip(value, min_val, max_val)
