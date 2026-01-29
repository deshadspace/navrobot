# simulation/robot/kinematics.py
import numpy as np

class DifferentialDriveKinematics:
    """Simple differential drive robot kinematics model"""
    
    def __init__(self, wheel_base=0.5, max_speed=1.0, max_angular_speed=np.pi/2):
        self.wheel_base = wheel_base
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
    
    def forward_kinematics(self, state, linear_vel, angular_vel, dt):
        """
        Update robot pose given velocities
        state: [x, y, theta]
        Returns: new_state [x, y, theta]
        """
        # Clip velocities
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        x, y, theta = state
        
        # Update pose
        x_new = x + linear_vel * np.cos(theta) * dt
        y_new = y + linear_vel * np.sin(theta) * dt
        theta_new = theta + angular_vel * dt
        
        # Normalize angle to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new])
    
    def inverse_kinematics(self, current_pos, target_pos):
        """
        Calculate velocities needed to move toward target
        Returns: (linear_vel, angular_vel)
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Desired heading
        desired_theta = np.arctan2(dy, dx)
        current_theta = current_pos[2] if len(current_pos) > 2 else 0
        
        # Angular error
        angle_error = np.arctan2(np.sin(desired_theta - current_theta),
                                 np.cos(desired_theta - current_theta))
        
        # Distance to target
        distance = np.sqrt(dx**2 + dy**2)
        
        # Simple proportional control
        linear_vel = min(distance * 0.5, self.max_speed)
        angular_vel = np.clip(angle_error * 2.0, -self.max_angular_speed, self.max_angular_speed)
        
        return linear_vel, angular_vel
