"""
Alert system for model monitoring.
"""

from typing import List, Dict
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule:
    """Define alert conditions."""
    
    def __init__(self, name: str, condition_fn, severity: AlertSeverity = AlertSeverity.WARNING):
        """
        Args:
            name: rule name
            condition_fn: function returning bool
            severity: alert severity
        """
        self.name = name
        self.condition_fn = condition_fn
        self.severity = severity
    
    def evaluate(self, metrics: Dict) -> Dict:
        """
        Evaluate rule against metrics.
        
        Args:
            metrics: metrics dict
            
        Returns:
            alert dict if triggered, None otherwise
        """
        try:
            if self.condition_fn(metrics):
                return {
                    "name": self.name,
                    "severity": self.severity.value,
                    "message": f"Alert: {self.name}",
                }
        except Exception:
            pass
        
        return None


class AlertSystem:
    """Manage system alerts."""
    
    def __init__(self):
        """Initialize alert system."""
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Dict] = []
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules.append(rule)
    
    def evaluate(self, metrics: Dict) -> List[Dict]:
        """
        Evaluate all rules.
        
        Args:
            metrics: metrics dict
            
        Returns:
            list of triggered alerts
        """
        new_alerts = []
        
        for rule in self.rules:
            alert = rule.evaluate(metrics)
            if alert:
                new_alerts.append(alert)
        
        self.active_alerts = new_alerts
        
        return new_alerts


# Common alert rules
ACCURACY_DROP = AlertRule(
    "accuracy_drop",
    lambda m: m.get("accuracy", 1.0) < 0.85,
    AlertSeverity.CRITICAL,
)

HIGH_LATENCY = AlertRule(
    "high_latency",
    lambda m: m.get("latency_ms", 0) > 100,
    AlertSeverity.WARNING,
)

HIGH_ERROR_RATE = AlertRule(
    "high_error_rate",
    lambda m: m.get("error_rate", 0) > 0.05,
    AlertSeverity.CRITICAL,
)
