"""
Model monitoring and drift detection.

Uses Evidently AI for data/prediction drift monitoring.
"""

import numpy as np
from typing import Dict, Tuple


class DriftDetector:
    """Detect data drift and model performance degradation."""
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        """
        Args:
            reference_data: baseline data distribution
            threshold: drift threshold (0.05 = 5% difference)
        """
        self.reference_mean = reference_data.mean()
        self.reference_std = reference_data.std()
        self.threshold = threshold
        self.drift_history = []
    
    def detect(self, new_data: np.ndarray) -> Dict[str, float]:
        """
        Detect data drift using statistical tests.
        
        Args:
            new_data: new data samples
            
        Returns:
            drift metrics
        """
        new_mean = new_data.mean()
        new_std = new_data.std()
        
        # Standardized difference
        mean_shift = abs(new_mean - self.reference_mean) / (self.reference_std + 1e-8)
        std_shift = abs(new_std - self.reference_std) / (self.reference_std + 1e-8)
        
        is_drift = mean_shift > self.threshold or std_shift > self.threshold
        
        metrics = {
            "mean_shift": float(mean_shift),
            "std_shift": float(std_shift),
            "is_drift": bool(is_drift),
        }
        
        self.drift_history.append(metrics)
        
        return metrics


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: number of predictions to track
        """
        self.window_size = window_size
        self.predictions = []
        self.actuals = []
        self.metrics_history = []
    
    def update(self, prediction: float, actual: float) -> None:
        """Add prediction/actual pair."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        # Keep window size
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute current metrics."""
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        # MAE
        mae = np.mean(np.abs(predictions - actuals))
        
        # Accuracy (for classification)
        accuracy = np.mean(predictions == actuals)
        
        return {
            "mae": float(mae),
            "accuracy": float(accuracy),
            "num_samples": len(self.predictions),
        }


class AlertManager:
    """Manage alerts for model issues."""
    
    def __init__(self, email: str = "admin@example.com"):
        """
        Args:
            email: email for alerts
        """
        self.email = email
        self.alerts = []
    
    def check_performance(self, metrics: Dict) -> None:
        """
        Check if metrics trigger alerts.
        
        Args:
            metrics: performance metrics
        """
        if metrics.get("mae", 0) > 0.5:
            self.create_alert("High prediction error", severity="warning")
        
        if metrics.get("accuracy", 1.0) < 0.8:
            self.create_alert("Low accuracy", severity="critical")
    
    def create_alert(self, message: str, severity: str = "info") -> None:
        """
        Create and log alert.
        
        Args:
            message: alert message
            severity: "info", "warning", "critical"
        """
        alert = {
            "message": message,
            "severity": severity,
            "timestamp": str(np.datetime64('now')),
        }
        
        self.alerts.append(alert)
        print(f"🚨 [{severity.upper()}] {message}")
