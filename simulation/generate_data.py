# simulation/generate_data.py
import os
import numpy as np
from PIL import Image
from simulation.envs.gridworld import GridWorldEnv

# Paths
os.makedirs("data/processed/images", exist_ok=True)
os.makedirs("data/processed/labels", exist_ok=True)

env = GridWorldEnv()
obs, _ = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)

    # Save image
    img_path = f"data/processed/images/img_{step}.png"
    Image.fromarray(obs).save(img_path)

    # Save label (here just example format for YOLO)
    # YOLO format: class x_center y_center width height (normalized 0-1)
    labels = []
    # obstacle
    for o in env.obstacles:
        x, y = o
        labels.append(f"0 {x/env.size} {y/env.size} 0.05 0.05")
    # human
    x, y = env.human_pos
    labels.append(f"1 {x/env.size} {y/env.size} 0.05 0.05")

    with open(f"data/processed/labels/img_{step}.txt", "w") as f:
        f.write("\n".join(labels))

    if done:
        break
