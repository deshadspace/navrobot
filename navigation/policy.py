"""
Policy inference and deployment.

Load trained policies and generate actions at inference time.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class PolicyInference:
    """Load and run trained RL policy."""
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        """
        Args:
            model_path: path to saved model
            device: "cpu" or "cuda"
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load policy model."""
        if self.model_path.suffix == ".pt":
            # PyTorch model
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        elif self.model_path.suffix == ".onnx":
            # ONNX model
            try:
                import onnxruntime as rt
                self.model = rt.InferenceSession(str(self.model_path))
            except ImportError:
                raise ImportError("onnxruntime required for ONNX models")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Generate action from observation.
        
        Args:
            observation: state observation (image or feature vector)
            deterministic: use mean action (True) or sample (False)
            
        Returns:
            action array [v_linear, v_angular]
        """
        if isinstance(self.model, torch.nn.Module):
            return self._torch_predict(observation, deterministic)
        else:
            return self._onnx_predict(observation)
    
    def _torch_predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """PyTorch inference."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            
            if deterministic:
                action = self.model(obs_tensor).mean
            else:
                distribution = self.model(obs_tensor)
                action = distribution.sample()
            
            action_np = action.cpu().numpy()[0]
        
        return action_np
    
    def _onnx_predict(self, observation: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        obs_batch = np.expand_dims(observation, axis=0).astype(np.float32)
        action = self.model.run([output_name], {input_name: obs_batch})[0]
        
        return action[0]


class RolloutBuffer:
    """Collect trajectories for offline RL."""
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: maximum number of transitions
        """
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_obs = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ):
        """Add transition to buffer."""
        if len(self.observations) >= self.capacity:
            # Remove oldest
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.next_obs.pop(0)
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_obs.append(next_obs)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of transitions."""
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        return {
            "observations": np.array([self.observations[i] for i in indices]),
            "actions": np.array([self.actions[i] for i in indices]),
            "rewards": np.array([self.rewards[i] for i in indices]),
            "dones": np.array([self.dones[i] for i in indices]),
            "next_obs": np.array([self.next_obs[i] for i in indices]),
        }
    
    def __len__(self) -> int:
        return len(self.observations)
