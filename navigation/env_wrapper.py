"""
Environment wrapper for RL compatibility.

Adapts simulation environment to stable-baselines3 interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional


class RLEnvironmentWrapper(gym.Env):
    """
    Wrapper making GridWorld compatible with RL algorithms.
    
    Converts observations to proper format and handles action/reward shaping.
    """
    
    def __init__(self, base_env: gym.Env, normalize_obs: bool = True):
        """
        Args:
            base_env: underlying simulation environment
            normalize_obs: normalize observations to [-1, 1]
        """
        self.base_env = base_env
        self.normalize_obs = normalize_obs
        
        # Observation space: camera image + state vector
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )
        
        # Action space: continuous velocity commands
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([0.5, 1.0]),
            dtype=np.float32,
        )
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.base_env.seed(seed)
        
        obs, info = self.base_env.reset(**kwargs)
        
        # Process observation
        obs = self._process_obs(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return observation, reward, dones, info.
        
        Args:
            action: [v_linear, v_angular]
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Execute in base environment
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        # Process observation
        obs = self._process_obs(obs)
        
        # Reward shaping can be applied here
        reward = self._shape_reward(reward, info)
        
        return obs, reward, done, truncated, info
    
    def _process_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Process observation from base environment."""
        if isinstance(obs, dict):
            # Extract image (camera observation)
            image = obs.get("image", obs.get("rgb", np.zeros((64, 64, 3), dtype=np.uint8)))
        else:
            image = obs
        
        if self.normalize_obs and image.max() > 1.0:
            image = image / 255.0
        
        return image.astype(np.uint8)
    
    def _shape_reward(self, reward: float, info: Dict) -> float:
        """
        Shape reward for better learning.
        
        Args:
            reward: original reward from environment
            info: info dict with context
            
        Returns:
            shaped reward
        """
        # Can add additional reward shaping here
        # Example: encourage forward progress
        
        return reward
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment."""
        return self.base_env.render()
    
    def close(self):
        """Close environment."""
        self.base_env.close()
