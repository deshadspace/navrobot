"""
Retrain pipeline.

Triggered on new data or performance degradation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class RetrainPipeline:
    """Orchestrate model retraining."""
    
    def __init__(self, config_path: Path = Path("configs/training.yaml")):
        """
        Args:
            config_path: path to training config
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def check_performance_drift(self) -> bool:
        """
        Check if deployed model has degraded.
        
        Returns:
            True if retraining needed
        """
        print("📈 Checking model performance...")
        
        # Placeholder: would fetch production metrics
        current_accuracy = 0.85
        baseline_accuracy = 0.90
        
        drift_detected = (baseline_accuracy - current_accuracy) > 0.05
        
        if drift_detected:
            print(f"⚠️ Performance drift detected: {current_accuracy:.2f} -> {baseline_accuracy:.2f}")
        
        return drift_detected
    
    def check_new_data(self) -> bool:
        """
        Check if new data available for training.
        
        Returns:
            True if new data exists
        """
        print("🔍 Checking for new data...")
        
        data_dir = Path(self.config["data"]["raw_data_path"])
        new_episodes = list(data_dir.glob("*.npz"))
        
        has_new_data = len(new_episodes) > 0
        
        if has_new_data:
            print(f"✓ Found {len(new_episodes)} new episodes")
        
        return has_new_data
    
    def should_retrain(self) -> bool:
        """Determine if retraining is necessary."""
        return self.check_performance_drift() or self.check_new_data()
    
    def run(self) -> Dict[str, Any]:
        """Execute retraining if needed."""
        print("🔄 Retrain pipeline starting...")
        
        if not self.should_retrain():
            print("✓ No retraining needed")
            return {"status": "skipped"}
        
        print("🚀 Triggering full retraining...")
        
        # Import and run training pipeline
        from pipelines.training_pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline()
        pipeline.run()
        
        return {"status": "completed"}
