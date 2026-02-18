"""Navigation README - Decision Making with RL"""

# Navigation Module - Decision Making & RL

## Purpose

The navigation module implements **reinforcement learning** for robot decision-making.

- Train policies with PPO, SAC, or DDPG
- Learn collision avoidance from camera feedback
- Deploy policies for real-time inference
- Monitor and improve performance over time

## Structure

```
navigation/
├── env_wrapper.py      # Gym-compatible environment adapter
├── reward.py           # Safety-aware reward functions
├── train_rl.py         # RL training entrypoint (PPO/SAC)
├── policy.py           # Inference and policy deployment
└── README.md
```

## Quick Start

### Train RL Policy

```bash
python navigation/train_rl.py \
  --algorithm PPO \
  --total_timesteps 1000000 \
  --batch_size 64 \
  --learning_rate 3e-4
```

### Evaluate Policy

```bash
python navigation/train_rl.py \
  --mode eval \
  --model_path checkpoints/navigation/policy_best.zip \
  --num_episodes 10
```

### Deploy Policy

```python
from navigation.policy import PolicyInference
import numpy as np

policy = PolicyInference("checkpoints/navigation/policy_best.zip")

# Run in environment
observation, _ = env.reset()
for _ in range(500):
    action = policy.predict(observation, deterministic=True)
    observation, reward, done, _, _ = env.step(action)
    if done:
        break
```

## RL Algorithms

### PPO (Proximal Policy Optimization)
**Best for**: Sample efficiency, stable training
```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
)
model.learn(total_timesteps=1e6)
```

**Hyperparameters**:
- `n_steps`: rollout length before update
- `n_epochs`: gradient updates per batch
- `clip_range`: PPO clipping parameter (0.2)
- `ent_coef`: entropy bonus (0.01)

### SAC (Soft Actor-Critic)
**Best for**: Off-policy learning, max entropy RL
```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1e6,
    learning_starts=10000,
    target_update_interval=1,
)
model.learn(total_timesteps=1e6)
```

**Advantages**:
- Off-policy (better sample efficiency)
- Automatic entropy tuning
- Good exploration/exploitation trade-off

## Training Pipeline

### 1. Data Collection
```python
from simulation.envs.gridworld import GridWorldEnv
from navigation.env_wrapper import RLEnvironmentWrapper

env = GridWorldEnv(grid_size=32, render=False)
env = RLEnvironmentWrapper(env)
```

### 2. Train Agent
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, learning_rate=3e-4)

model.learn(
    total_timesteps=1e6,
    callback=[CheckpointCallback(...), EvalCallback(...)],
    progress_bar=True,
)
```

### 3. Evaluate
```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

### 4. Deploy
```python
policy = PolicyInference("checkpoints/policy_best.zip")
```

## Reward Function

Safety-aware rewards balance task completion with safety:

```python
from navigation.reward import SafetyAwareReward

reward_fn = SafetyAwareReward(
    goal_reach_reward=10.0,      # reaching goal
    collision_penalty=-1.0,       # collision avoidance
    step_penalty=-0.01,          # efficiency
    progress_reward_scale=0.1,   # moving toward goal
)

reward = reward_fn(
    done=done,
    collision=collision,
    goal_dist_prev=prev_dist,
    goal_dist_curr=curr_dist,
    info=info,
)
```

## Performance Metrics

Track training with these metrics:

- **Success Rate**: % of episodes reaching goal
- **Collision Rate**: % of steps with collision
- **Average Reward**: mean episode return
- **Path Length**: steps to goal (efficiency)
- **Training Stability**: variance in returns

## Tensorboard Monitoring

```bash
# During training (callbacks log automatically)
tensorboard --logdir logs/navigation

# View in browser
# http://localhost:6006
```

## Troubleshooting

### Training Not Converging
```python
# Reduce learning rate
model.learning_rate = 1e-4

# Increase entropy bonus (encourage exploration)
model.ent_coef = 0.05

# Check reward scale
print(f"Episode rewards: {episode_rewards}")
```

### High Collision Rate
```python
# Increase collision penalty
reward_fn.collision_penalty = -5.0

# Use expert demonstrations (behavioral cloning first)
from imitation.algorithms import BC

bc_trainer = BC(...)
```

### Sample Inefficiency
```python
# Use off-policy SAC instead
model = SAC("MlpPolicy", env, buffer_size=int(1e6))

# or: Use experience replay with PPO
model = PPO(..., n_steps=4096)  # larger batches
```

## Production Deployment

### Export Policy
```bash
python -c "
from navigation.policy import PolicyInference

policy = PolicyInference('checkpoints/policy.zip')
policy.export_onnx('models/policy.onnx')
"
```

### Real-time Inference
```python
import numpy as np
from navigation.policy import PolicyInference

policy = PolicyInference("models/policy.onnx", device="cpu")

while True:
    obs = camera.capture()  # get image
    action = policy.predict(obs, deterministic=True)
    robot.execute_action(action)
```

### A/B Testing
```python
import random
from navigation.policy import PolicyInference

policy_v1 = PolicyInference("models/policy_v1.onnx")
policy_v2 = PolicyInference("models/policy_v2.onnx")

policy = policy_v1 if random.random() < 0.5 else policy_v2
```

## Continuous Learning

Retrain with collected data:

```bash
# 1. Collect data with deployed policy
python scripts/collect_rollouts.py --num_episodes 100

# 2. Add to training buffer
dvc add data/rollouts.tar.gz

# 3. Retrain
python navigation/train_rl.py --continue_training

# 4. Evaluate vs current policy
# If better → deploy new version
```

## Next Steps

- [ ] Implement imitation learning from expert demos
- [ ] Multi-task RL (navigation + manipulation)
- [ ] Domain randomization for sim-to-real transfer
- [ ] Curiosity-driven exploration
- [ ] Hierarchical RL (high-level planner + low-level controller)
- [ ] Multi-agent scenarios
