"""
Monitoring utilities and metric collection.
"""

from typing import Dict, List
import time


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        """Initialize collector."""
        self.metrics = {}
    
    def record(self, name: str, value: float) -> None:
        """
        Record metric value.
        
        Args:
            name: metric name
            value: metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": time.time(),
        })
    
    def get_stats(self, name: str) -> Dict:
        """
        Get statistics for metric.
        
        Args:
            name: metric name
            
        Returns:
            statistics dict
        """
        import numpy as np
        
        if name not in self.metrics:
            return {}
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()


class LatencyTracker:
    """Track inference latency."""
    
    def __init__(self):
        """Initialize tracker."""
        self.latencies = []
    
    def record(self, latency_ms: float) -> None:
        """Record latency in milliseconds."""
        self.latencies.append(latency_ms)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        import numpy as np
        
        if not self.latencies:
            return {}
        
        p_values = [50, 95, 99]
        percentiles = {}
        
        for p in p_values:
            percentiles[f"p{p}"] = float(np.percentile(self.latencies, p))
        
        return percentiles
