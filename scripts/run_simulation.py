# scripts/run_simulation.py
from simulation.envs.gridworld import GridWorldEnv

env = GridWorldEnv()
obs, _ = env.reset()

for step in range(20):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Step {step}, Collision: {info['collision']}")
    if done:
        break
