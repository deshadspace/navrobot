# simulation/envs/gridworld.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class GridWorldEnv(gym.Env):
    """Enhanced GridWorld environment for robot navigation"""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size=10, grid_size=64, render_mode=None):
        self.size = size  # world size in meters
        self.grid_size = grid_size  # pixel resolution
        self.render_mode = render_mode
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi/2]),
            high=np.array([1.0, np.pi/2]),
            dtype=np.float32
        )
        
        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8
        )
        
        self.dt = 0.1  # time step
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize robot pose [x, y, theta]
        self.robot_pos = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        
        # Goal position
        self.goal_pos = np.array([self.size-1.5, self.size-1.5], dtype=np.float32)
        
        # Human position (dynamic obstacle)
        self.human_pos = np.array([self.size/2, self.size/2], dtype=np.float32)
        self.human_vel = np.array([0.2, 0.1], dtype=np.float32)
        
        # Static obstacles
        self.obstacles = [
            np.array([3.0, 3.0], dtype=np.float32),
            np.array([5.0, 5.0], dtype=np.float32),
            np.array([7.0, 3.0], dtype=np.float32)
        ]
        
        self.steps = 0
        self.max_steps = 500
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        self.steps += 1
        
        # Update robot position
        linear_vel, angular_vel = action
        
        # Simple kinematics
        x, y, theta = self.robot_pos
        x += linear_vel * np.cos(theta) * self.dt
        y += linear_vel * np.sin(theta) * self.dt
        theta += angular_vel * self.dt
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # normalize
        
        self.robot_pos = np.array([x, y, theta])
        
        # Keep robot in bounds
        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.size)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.size)
        
        # Update human position (simple linear motion)
        self.human_pos += self.human_vel * self.dt
        
        # Bounce human off walls
        if self.human_pos[0] <= 0 or self.human_pos[0] >= self.size:
            self.human_vel[0] *= -1
        if self.human_pos[1] <= 0 or self.human_pos[1] >= self.size:
            self.human_vel[1] *= -1
        
        # Check collisions
        collision = self._check_collision()
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.robot_pos[:2] - self.goal_pos)
        goal_reached = dist_to_goal < 0.5
        
        # Calculate reward
        reward = self._compute_reward(collision, goal_reached, dist_to_goal)
        
        # Episode termination
        terminated = collision or goal_reached
        truncated = self.steps >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        info['collision'] = collision
        info['goal_reached'] = goal_reached
        info['distance_to_goal'] = dist_to_goal
        
        return obs, reward, terminated, truncated, info

    def _check_collision(self):
        """Check if robot collides with obstacles or human"""
        robot_pos_2d = self.robot_pos[:2]
        collision_radius = 0.4
        
        # Check static obstacles
        for obs in self.obstacles:
            if np.linalg.norm(robot_pos_2d - obs) < collision_radius:
                return True
        
        # Check human
        if np.linalg.norm(robot_pos_2d - self.human_pos) < collision_radius:
            return True
        
        return False

    def _compute_reward(self, collision, goal_reached, dist_to_goal):
        """Calculate reward based on current state"""
        if collision:
            return -100.0
        if goal_reached:
            return 100.0
        
        # Distance-based reward (encourage moving toward goal)
        reward = -dist_to_goal * 0.1
        
        # Small step penalty
        reward -= 0.01
        
        return reward

    def _get_obs(self):
        """Generate RGB observation"""
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * 255
        
        scale = self.grid_size / self.size
        
        # Draw obstacles (black)
        for obs in self.obstacles:
            center = (int(obs[0] * scale), int(obs[1] * scale))
            cv2.circle(img, center, int(0.3 * scale), (0, 0, 0), -1)
        
        # Draw human (red)
        human_center = (int(self.human_pos[0] * scale), int(self.human_pos[1] * scale))
        cv2.circle(img, human_center, int(0.3 * scale), (0, 0, 255), -1)
        
        # Draw goal (green)
        goal_center = (int(self.goal_pos[0] * scale), int(self.goal_pos[1] * scale))
        cv2.circle(img, goal_center, int(0.4 * scale), (0, 255, 0), -1)
        
        # Draw robot (blue) with orientation
        robot_center = (int(self.robot_pos[0] * scale), int(self.robot_pos[1] * scale))
        cv2.circle(img, robot_center, int(0.3 * scale), (255, 0, 0), -1)
        
        # Draw orientation line
        theta = self.robot_pos[2]
        end_x = int((self.robot_pos[0] + 0.4 * np.cos(theta)) * scale)
        end_y = int((self.robot_pos[1] + 0.4 * np.sin(theta)) * scale)
        cv2.line(img, robot_center, (end_x, end_y), (255, 255, 0), 2)
        
        return img

    def _get_info(self):
        """Return additional information"""
        return {
            'robot_pos': self.robot_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'human_pos': self.human_pos.copy(),
            'steps': self.steps
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            img = self._get_obs()
            cv2.imshow("GridWorld", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
