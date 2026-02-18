"""
Simulation README - Data Factory & Physics Engine

## Purpose

The simulation module is the **data factory** for the entire ML pipeline.

- Generates diverse, labeled episodes for RL training
- Implements physics and sensor models
- Supports both simple (GridWorld) and complex environments

## Structure

```
simulation/
├── envs/
│   ├── gridworld.py          # Custom Gym environment
│   └── __init__.py
├── sensors/
│   ├── camera.py             # RGB rendering
│   └── lidar.py              # Range sensing
├── robot/
│   ├── controller.py         # Low-level control loop
│   └── kinematics.py         # Motion model
├── generate_data.py          # Batch episode generation
└── README.md
```

## Quick Start

### Generate Simulation Data

```bash
python simulation/generate_data.py \
  --num_episodes 1000 \
  --max_steps 500 \
  --output data/raw/simulation_logs/
```

### Run Single Episode

```bash
from simulation.envs.gridworld import GridWorldEnv

env = GridWorldEnv(render=True)
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

env.close()
```

### Visualize Environment

```bash
python -c "
from simulation.envs.gridworld import GridWorldEnv
env = GridWorldEnv(render=True)
env.reset()
for _ in range(100):
    env.step(env.action_space.sample())
env.close()
"
```

## Environment Details

### GridWorld-v1

- **Observation Space**:
  - `image`: RGB camera [64, 64, 3]
  - `state`: [x, y, theta, v_lin, v_ang] normalized
  
- **Action Space**:
  - Continuous [v_linear, v_angular]
  - Bounds: linear ∈ [-0.5, 0.5] m/s, angular ∈ [-1.0, 1.0] rad/s

- **Rewards**:
  - Goal reached: +10.0
  - Step: -0.01
  - Collision: -1.0
  - Out of bounds: -1.0

### Sensor Models

#### Camera
- Resolution: 64×64 RGB
- FOV: 90°
- Simulated ray-casting with occlusion

#### LiDAR (optional)
- 32 rays
- Range: 8.0 meters
- Scan frequency: 10 Hz

## Scenarios

The data generator supports multiple navigation scenarios:

1. **Point Goal**: Reach random waypoint
2. **Trajectory Tracking**: Follow predefined path
3. **Obstacle Avoidance**: Navigate cluttered scenes
4. **Multi-Goal**: Sequential waypoints

## Physics & Realism

- **Kinematics**: Differential-drive robot (Bicycle model)
- **Dynamics**: Simplified (no slip, infinite torque)
- **Collision Detection**: Circle-based for speed
- **Sampling**: 10 Hz control, 100 Hz physics

## Extending the Simulation

### Add Custom Environment

```python
from gymnasium import Env
import numpy as np

class MyEnv(Env):
    def __init__(self):
        self.observation_space = ...
        self.action_space = ...
    
    def reset(self, seed=None):
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        # Apply physics
        reward = self._compute_reward()
        done = self._check_termination()
        return obs, reward, done, False, {}
```

### Add Custom Sensor

```python
class RGBDCamera:
    def render(self, world_state, robot_pose):
        rgb = ...  # render RGB
        depth = ...  # compute depth
        return np.concatenate([rgb, depth], axis=2)
```

## Performance

- **Episode Generation**: ~100 episodes/min (CPU)
- **Rendering**: 30 FPS headless
- **Memory**: <100 MB per 1000 episodes

## Future Work

- [ ] 3D environment with physics engine (PyBullet)
- [ ] Photorealistic rendering (Gazebo → ROS Bridge)
- [ ] Real-to-sim domain randomization
- [ ] Multi-robot scenarios
- [ ] Partial observability (sensor occlusion)
