# simulation/robot/controller.py
import numpy as np

class ObstacleAvoidanceController:
    """Simple reactive obstacle avoidance controller"""
    
    def __init__(self, safe_distance=1.0, detection_range=3.0):
        self.safe_distance = safe_distance
        self.detection_range = detection_range
    
    def compute_action(self, robot_pos, target_pos, obstacles):
        """
        Compute control action to avoid obstacles and reach target
        
        Args:
            robot_pos: [x, y, theta] current robot pose
            target_pos: [x, y] goal position
            obstacles: list of [x, y] obstacle positions
            
        Returns:
            action: [linear_vel, angular_vel]
        """
        # Calculate attractive force toward goal
        goal_force = self._attractive_force(robot_pos[:2], target_pos)
        
        # Calculate repulsive forces from obstacles
        repulsive_force = np.zeros(2)
        for obs in obstacles:
            force = self._repulsive_force(robot_pos[:2], obs)
            repulsive_force += force
        
        # Combine forces
        total_force = goal_force + repulsive_force
        
        # Convert to velocity commands
        linear_vel = min(np.linalg.norm(total_force), 1.0)
        
        # Calculate desired heading
        if np.linalg.norm(total_force) > 0.01:
            desired_theta = np.arctan2(total_force[1], total_force[0])
            current_theta = robot_pos[2] if len(robot_pos) > 2 else 0
            angle_error = np.arctan2(np.sin(desired_theta - current_theta),
                                    np.cos(desired_theta - current_theta))
            angular_vel = np.clip(angle_error * 2.0, -np.pi/2, np.pi/2)
        else:
            angular_vel = 0.0
        
        return np.array([linear_vel, angular_vel])
    
    def _attractive_force(self, pos, target):
        """Attractive force pulling robot toward goal"""
        direction = target - pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:
            return np.zeros(2)
        
        # Proportional attraction
        force_magnitude = min(distance * 0.5, 1.0)
        return (direction / distance) * force_magnitude
    
    def _repulsive_force(self, pos, obstacle):
        """Repulsive force pushing robot away from obstacle"""
        direction = pos - obstacle
        distance = np.linalg.norm(direction)
        
        if distance > self.detection_range or distance < 0.01:
            return np.zeros(2)
        
        # Inverse square law repulsion
        force_magnitude = min(
            (self.safe_distance / distance) ** 2,
            2.0
        )
        
        return (direction / distance) * force_magnitude

class PIDController:
    """Simple PID controller for trajectory tracking"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, error, dt):
        """Compute control output"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
