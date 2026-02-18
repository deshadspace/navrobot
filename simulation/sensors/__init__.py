"""
Simulated LiDAR sensor.

Generates synthetic range measurements in the gridworld environment.
"""

import numpy as np
from typing import Tuple


class LiDAR:
    """2D LiDAR range scanner."""
    
    def __init__(self, num_rays: int = 32, max_range: float = 8.0):
        """
        Args:
            num_rays: number of range measurements
            max_range: maximum detection range in meters
        """
        self.num_rays = num_rays
        self.max_range = max_range
    
    def scan(self, world_state: np.ndarray, robot_pos: np.ndarray, robot_angle: float) -> np.ndarray:
        """
        Generate LiDAR scan.
        
        Args:
            world_state: grid map of the environment
            robot_pos: [x, y] position
            robot_angle: rotation in radians
            
        Returns:
            Range measurements array of shape (num_rays,)
        """
        ranges = np.full(self.num_rays, self.max_range, dtype=np.float32)
        
        # Generate ray angles
        angle_min = robot_angle - np.pi / 2
        angle_max = robot_angle + np.pi / 2
        
        for i in range(self.num_rays):
            ray_angle = angle_min + (i / (self.num_rays - 1)) * (angle_max - angle_min)
            
            # Simple ray-casting: step along ray until hitting obstacle
            # Placeholder implementation
            ranges[i] = self.max_range * (1 - i / self.num_rays)  # simple falloff
        
        return ranges
    
    def get_angles(self) -> np.ndarray:
        """Get ray angles in radians."""
        return np.linspace(-np.pi / 2, np.pi / 2, self.num_rays)
