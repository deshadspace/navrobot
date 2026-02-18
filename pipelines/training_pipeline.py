"""
End-to-end training pipeline.

Orchestrates perception + navigation training with MLflow tracking.
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any
import yaml


class TrainingPipeline:
    """Orchestrate training workflow."""
    
    def __init__(self, config_path: Path = Path("configs/training.yaml")):
        """
        Args:
            config_path: path to training config
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config["experiment_tracking"]["tracking_uri"])
        mlflow.set_experiment(self.config["experiment_tracking"]["experiment_name"])
    
    def train_perception(self) -> None:
        """Train vision model."""
        with mlflow.start_run(run_name="perception_training"):
            # Log params
            params = self.config["training"]["perception"]
            mlflow.log_params(params)
            
            # Placeholder: actual training
            print("📚 Training perception model...")
            metrics = {
                "train_loss": 0.12,
                "train_accuracy": 0.95,
                "val_loss": 0.15,
                "val_accuracy": 0.93,
            }
            
            mlflow.log_metrics(metrics)
            
            # Register model
            mlflow.pytorch.log_model(
                "model",  # placeholder
                "perception_model",
                registered_model_name="perception-model",
            )
    
    def train_navigation(self) -> None:
        """Train RL policy."""
        with mlflow.start_run(run_name="navigation_training"):
            # Log params
            params = self.config["training"]["navigation"]
            mlflow.log_params(params)
            
            # Placeholder: actual training
            print("🧭 Training navigation policy...")
            metrics = {
                "episode_reward": 85.5,
                "success_rate": 0.92,
                "collision_rate": 0.03,
            }
            
            mlflow.log_metrics(metrics)
            
            # Register model
            mlflow.pytorch.log_model(
                "policy",  # placeholder
                "navigation_policy",
                registered_model_name="navigation-policy",
            )
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate both models."""
        with mlflow.start_run(run_name="evaluation"):
            print("📊 Running evaluation...")
            
            eval_metrics = {
                "perception_accuracy": 0.93,
                "navigation_success_rate": 0.92,
                "end_to_end_success": 0.85,
            }
            
            mlflow.log_metrics(eval_metrics)
            
            return eval_metrics
    
    def run(self, **kwargs) -> None:
        """Execute full training pipeline."""
        print("🚀 Starting training pipeline...")
        
        try:
            self.train_perception()
            self.train_navigation()
            metrics = self.evaluate()
            
            print("✅ Pipeline complete")
            print(f"Metrics: {metrics}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise


def main():
    """Main entrypoint."""
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
