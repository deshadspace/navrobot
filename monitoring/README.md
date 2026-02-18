"""Monitoring README - Model Trust & Safety

## Purpose

The monitoring module ensures **model reliability in production**.

- Detect data/model drift automatically
- Track performance metrics over time
- Alert on degradation
- Trigger retraining when needed

## Structure

```
monitoring/
├── drift.py         # Statistical drift detection
├── metrics.py       # Metrics collection & aggregation
├── alerts.py        # Alert rules & system
└── README.md
```

## Data Drift Detection

Detect when input data distribution changes:

```python
from monitoring.drift import DriftDetector
import numpy as np

# Initialize with reference data
reference_data = np.random.randn(1000)
detector = DriftDetector(reference_data, threshold=0.05)

# Check new data
new_data = np.random.randn(100)
drift_metrics = detector.detect(new_data)

print(drift_metrics)
# {'mean_shift': 0.12, 'std_shift': 0.08, 'is_drift': True}
```

**Action**: If drift detected → trigger retraining

## Performance Monitoring

Track prediction accuracy over time:

```python
from monitoring.drift import PerformanceMonitor

monitor = PerformanceMonitor(window_size=100)

# During inference
for pred, actual in predictions:
    monitor.update(pred, actual)

metrics = monitor.get_metrics()
print(metrics)
# {'mae': 0.15, 'accuracy': 0.92, 'num_samples': 100}
```

## Alert System

Define and trigger alerts:

```python
from monitoring.alerts import AlertSystem, AlertRule, AlertSeverity

alert_sys = AlertSystem()

# Add rules
alert_sys.add_rule(AlertRule(
    "accuracy_drop",
    lambda m: m.get("accuracy") < 0.85,
    AlertSeverity.CRITICAL,
))

# Evaluate
alerts = alert_sys.evaluate({"accuracy": 0.82})
# [{'name': 'accuracy_drop', 'severity': 'critical', ...}]
```

## Integration with API

Monitor serving endpoint:

```python
from monitoring.metrics import MetricsCollector, LatencyTracker
import time

collector = MetricsCollector()
latency_tracker = LatencyTracker()

# In FastAPI endpoint
@app.post("/predict")
async def predict(request):
    start = time.time()
    
    result = model.predict(request.image)
    
    latency_ms = (time.time() - start) * 1000
    latency_tracker.record(latency_ms)
    collector.record("predictions", 1)
    
    return result
```

## Prometheus Metrics

Export metrics for monitoring dashboards:

```python
# In app.py
from prometheus_client import Counter, Histogram

predictions_total = Counter(
    'predictions_total',
    'Total predictions',
    ['model'],
)

prediction_latency = Histogram(
    'prediction_latency_ms',
    'Prediction latency',
)

@app.post("/predict")
async def predict(request):
    start = time.time()
    result = model.predict(request)
    
    predictions_total.labels(model='perception').inc()
    prediction_latency.observe((time.time() - start) * 1000)
    
    return result

# Scrape endpoint
@app.get("/metrics")
async def metrics():
    return generate_latest()
```

## Grafana Dashboard

View metrics:

```bash
# Start Prometheus
docker run -p 9090:9090 -v prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# Start Grafana
docker run -p 3000:3000 grafana/grafana

# Access dashboard at http://localhost:3000
```

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'robot-api'
    static_configs:
      - targets: ['localhost:8000']
```

## Monitoring Checklist

- [ ] Track prediction latency (p50, p95, p99)
- [ ] Monitor accuracy on holdout test set
- [ ] Detect input data drift
- [ ] Detect model output distribution shifts
- [ ] Alert on performance drop >5%
- [ ] Trigger retraining on drift detection
- [ ] Log all predictions for analysis
- [ ] A/B test new models before deployment

## Retraining Trigger

```python
from pipelines.retrain_pipeline import RetrainPipeline

if drift_detector.detect(new_data)["is_drift"]:
    print("🔄 Drift detected, starting retraining...")
    
    retrain = RetrainPipeline()
    result = retrain.run()
    
    if result["status"] == "completed":
        print("✓ Retraining complete, deploying new model...")
        deploy_new_model()
```

## Next Steps

- [ ] Implement Evidently AI for advanced drift detection
- [ ] Setup continuous monitoring dashboard
- [ ] Add feature importance tracking
- [ ] Implement model versioning in registry
- [ ] Create automated retraining workflows
- [ ] Add budget/resource alerts
