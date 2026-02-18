"""
Perception evaluation utilities.

Metrics for vision model assessment.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    _, pred_classes = predictions.max(dim=1)
    correct = (pred_classes == targets).sum().item()
    return correct / len(targets)


def compute_precision_recall(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute precision and recall."""
    _, pred_classes = predictions.max(dim=1)
    
    tp = ((pred_classes == 1) & (targets == 1)).sum().item()
    fp = ((pred_classes == 1) & (targets == 0)).sum().item()
    fn = ((pred_classes == 0) & (targets == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {"precision": precision, "recall": recall}


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    accuracy = compute_accuracy(predictions, targets)
    pr = compute_precision_recall(predictions, targets)
    
    f1 = 2 * (pr["precision"] * pr["recall"]) / (pr["precision"] + pr["recall"] + 1e-8)
    
    return {
        "accuracy": accuracy,
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": f1,
    }
