"""
Reward function for safe navigation.

Implements task-specific rewards with safety constraints.
"""

import numpy as np
from typing import Dict, Tuple


class SafetyAwareReward:
    """
    Reward function that balances task completion with safety.
    
    Encourages reaching goals while penalizing unsafe behaviors.
    """
    
    def __init__(
        self,
        goal_reach_reward: float = 10.0,
        collision_penalty: float = -1.0,
        step_penalty: float = -0.01,
        progress_reward_scale: float = 0.1,
    ):
        """
        Args:
            goal_reach_reward: bonus for reaching goal
            collision_penalty: penalty for collision
            step_penalty: penalty per step (encourages efficiency)
            progress_reward_scale: bonus scale for moving toward goal
        """
        self.goal_reach_reward = goal_reach_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.progress_reward_scale = progress_reward_scale
    
    def __call__(
        self,
        done: bool,
        collision: bool,
        goal_dist_prev: float,
        goal_dist_curr: float,
        info: Dict,
    ) -> float:
        """
        Compute reward.
        
        Args:
            done: episode terminated
            collision: collision occurred
            goal_dist_prev: distance to goal before step
            goal_dist_curr: distance to goal after step
            info: additional info dict
            
        Returns:
            reward signal
        """
        reward = 0.0
        
        # Goal reached
        if done and not collision:
            reward += self.goal_reach_reward
        
        # Collision penalty
        if collision:
            reward += self.collision_penalty
        
        # Step penalty (encourages efficiency)
        reward += self.step_penalty
        
        # Progress reward (encourage moving toward goal)
        progress = goal_dist_prev - goal_dist_curr
        if progress > 0:
            reward += progress * self.progress_reward_scale
        
        return reward
    
    @staticmethod
    def compute_metrics(trajectory: list) -> Dict[str, float]:
        """
        Compute summary metrics for episode.
        
        Args:
            trajectory: list of (obs, action, reward, done, info) tuples
            
        Returns:
            dict with metrics
        """
        total_reward = sum(r for _, _, r, _, _ in trajectory)
        num_steps = len(trajectory)
        
        collisions = sum(1 for _, _, _, _, info in trajectory if info.get("collision", False))
        
        return {
            "total_reward": total_reward,
            "num_steps": num_steps,
            "avg_reward": total_reward / num_steps,
            "collision_count": collisions,
            "collision_rate": collisions / num_steps,
        }


class DistributedReward:
    """Multi-task reward combining navigation + perception."""
    
    def __init__(self, nav_weight: float = 0.8, perception_weight: float = 0.2):
        """
        Args:
            nav_weight: weight for navigation reward
            perception_weight: weight for perception reward
        """
        self.nav_weight = nav_weight
        self.perception_weight = perception_weight
        self.nav_reward = SafetyAwareReward()
    
    def __call__(
        self,
        nav_reward: float,
        perception_info: Dict,
    ) -> float:
        """
        Combine navigation and perception rewards.
        
        Args:
            nav_reward: from SafetyAwareReward
            perception_info: dict with perception metrics
            
        Returns:
            combined reward
        """
        perception_reward = perception_info.get("detection_confidence", 0.0)
        
        total = (
            self.nav_weight * nav_reward +
            self.perception_weight * perception_reward
        )
        
        return total
