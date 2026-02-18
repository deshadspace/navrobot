#!/bin/bash

# Reproduce entire experiment from scratch
# Usage: ./scripts/reproduce_experiment.sh

set -e

echo "🔄 Reproducing full experiment..."

# 1. Generate data
echo "Step 1: Generating simulation data..."
python simulation/generate_data.py \
    --num_episodes 100 \
    --max_steps 500 \
    --output data/raw/simulation_logs/

# 2. Preprocess data
echo "Step 2: Preprocessing data..."
python -c "
from pipelines.data_pipeline import DataPipeline
pipeline = DataPipeline()
pipeline.run()
"

# 3. Train perception model
echo "Step 3: Training perception model..."
python perception/train.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3

# 4. Train navigation policy
echo "Step 4: Training navigation policy..."
python navigation/train_rl.py \
    --algorithm PPO \
    --total_timesteps 1000000 \
    --batch_size 64

# 5. Evaluate
echo "Step 5: Evaluating models..."
python -c "
from pipelines.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline()
metrics = pipeline.evaluate()
print(f'Final metrics: {metrics}')
"

# 6. Start API
echo "Step 6: Starting API server..."
echo "Run: ./scripts/start_server.sh"

echo "✅ Experiment reproduction complete!"
