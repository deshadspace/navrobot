"""
Simulated camera sensor.

Generates synthetic RGB images from the robot's perspective in the gridworld.
"""

import numpy as np
from typing import Tuple


class Camera:
    """Perspective camera for robot observation."""
    
    def __init__(self, resolution: Tuple[int, int] = (64, 64), fov: float = 90.0):
        """
        Args:
            resolution: (height, width) in pixels
            fov: field of view in degrees
        """
        self.resolution = resolution
        self.fov = fov
    
    def render(self, world_state: np.ndarray, robot_pos: np.ndarray, robot_angle: float) -> np.ndarray:
        """
        Generate RGB image from robot's viewpoint.
        
        Args:
            world_state: grid map of the environment
            robot_pos: [x, y] position
            robot_angle: rotation in radians
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        h, w = self.resolution
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple placeholder: render obstacles in view frustum
        # In production: ray-cast or full 3D renderer
        
        # Add gradient based on distance (depth perception)
        for y in range(h):
            depth = 1.0 - (y / h)  # far to near
            image[y, :] = int(255 * depth)
        
        return image
    
    def get_fov_radians(self) -> float:
        """Get FOV in radians."""
        return np.radians(self.fov)
