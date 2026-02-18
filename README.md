# Robot MLOps Autonomy 🤖

Production-ready MLOps + RL pipeline for autonomous navigation. Runs anywhere: local CPU, Colab, or cloud.

## 🎯 Overview

Complete end-to-end system for training autonomous navigation agents:
- **Simulation**: Generate diverse training data (GridWorld environment)
- **Perception**: Vision models for object detection/segmentation
- **Navigation**: RL policies (PPO/SAC) for collision-free navigation
- **Serving**: FastAPI inference service
- **Monitoring**: Automatic drift detection & retraining

Designed for **real MLOps teams** — not just toy examples.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Phases](#phases)
- [Installation](#installation)
- [Usage](#usage)

## 🚀 Quick Start

### Local Setup (5 min)

```bash
# Clone repo
git clone https://github.com/deshadspace/navrobot
cd navrobot

# Create environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Generate data
python simulation/generate_data.py --num_episodes 100

# Train perception model
python perception/train.py --epochs 50

# Train navigation policy
python navigation/train_rl.py --algorithm PPO --total_timesteps 100000

# Start API
uvicorn serving.app:app --reload
```

## 📁 Project Structure

```
robot-mlops-autonomy/
├── configs/                   # Configuration files (YAML)
│   ├── env.yaml              # Simulation config
│   ├── model.yaml            # Model hyperparameters
│   ├── training.yaml         # Training pipeline
│   └── serving.yaml          # API config
├── data/                      # Data directory (DVC-tracked)
│   ├── raw/                  # Original simulation data
│   ├── processed/            # Cleaned + resized data
│   └── features/             # Embeddings + state vectors
├── simulation/                #  Phase 1: Data Factory
├── perception/                #  Phase 2: Vision ML
├── navigation/                #  Phase 3: RL Decision Making
├── pipelines/                 #  Phase 4: MLOps Orchestration
├── serving/                   #  Phase 5: Production API
├── monitoring/                #  Phase 6: Trust & Safety
├── tests/                     # Test suite
├── experiments/               # Notebooks + results
├── scripts/                   # Utility scripts
├── ci_cd/                     # GitHub Actions workflows
├── pyproject.toml            # Project metadata
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🧭 Phases

**Phase 1: Simulation & Data** (Week 1)
```bash
python simulation/generate_data.py
```

**Phase 2: Perception Training** (Week 2)
```bash
python perception/train.py --epochs 50
```

**Phase 3: Navigation with RL** (Week 3)
```bash
python navigation/train_rl.py --algorithm PPO --total_timesteps 1000000
```

**Phase 4: Pipelines & MLOps** (Week 4)
```bash
python pipelines/training_pipeline.py
```

**Phase 5: Serving** (Week 5)
```bash
uvicorn serving.app:app --reload
```

**Phase 6: Monitoring** (Week 6)
```bash
python -c "from pipelines.retrain_pipeline import RetrainPipeline; RetrainPipeline().run()"
```

## 💻 Installation

### Requirements
- Python ≥ 3.10
- pip or conda
- ~2GB disk space

### From Source

```bash
git clone https://github.com/deshadspace/navrobot
cd navrobot
pip install -r requirements.txt
```

## 📖 Usage

### Generate Simulation Data

```python
from simulation.envs.gridworld import GridWorldEnv

env = GridWorldEnv(grid_size=32)
obs, info = env.reset()

for step in range(500):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, _ = env.reset()
```

### Train Perception Model

```bash
python perception/train.py --epochs 50 --batch_size 32
```

### Train RL Policy

```python
from stable_baselines3 import PPO
from navigation.env_wrapper import RLEnvironmentWrapper
from simulation.envs.gridworld import GridWorldEnv

env = GridWorldEnv()
env = RLEnvironmentWrapper(env)

model = PPO("MlpPolicy", env, learning_rate=3e-4)
model.learn(total_timesteps=1000000)
model.save("checkpoints/policy")
```

### Deploy API

```bash
uvicorn serving.app:app --reload
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📚 Documentation

- [Simulation README](simulation/README.md) - Environment and data generation
- [Perception README](perception/README.md) - Vision models and training
- [Navigation README](navigation/README.md) - RL policies and algorithms
- [Monitoring README](monitoring/README.md) - Drift detection and retraining

##  Contributing

Contributions welcome! Please fork and submit a pull request.

##  License

MIT License

##  Acknowledgments

Built with PyTorch, Stable-Baselines3, FastAPI, MLflow, and Gymnasium.

---

**Made with ❤️ for the robotics & MLOps community**
