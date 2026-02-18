"""
Perception Module README - Vision Learning

## Purpose

The perception module implements **computer vision** for the robot.

- Detect and classify objects in camera images
- Extract visual features for navigation decisions
- Train and evaluate vision models
- Export models for production inference

## Structure

```
perception/
├── datasets/
│   ├── __init__.py
│   └── vision_dataset.py     # PyTorch Dataset
├── models/
│   ├── __init__.py
│   ├── cnn_baseline.py       # Simple CNN + ResNet
│   └── yolo/                 # YOLO detector (future)
├── train.py                  # Training entrypoint
├── evaluate.py               # Metrics & evaluation
├── export.py                 # ONNX / TorchScript
└── README.md
```

## Quick Start

### Train Vision Model

```bash
python perception/train.py \
  --data_dir data/processed/images \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-3
```

### Evaluate Model

```bash
python perception/evaluate.py \
  --model_path checkpoints/perception/best_model.pt \
  --test_data data/processed/images/test
```

### Export for Production

```bash
python perception/export.py \
  --model_path checkpoints/perception/best_model.pt \
  --format onnx \
  --output_path models/perception_v1.onnx
```

## Models

### CNN Baseline
- **Architecture**: 3-layer CNN + fully connected head
- **Parameters**: ~500K
- **Speed**: <10ms per image (CPU)
- **Use case**: Lightweight mobile/edge inference

### ResNet18
- **Architecture**: ResNet-18 (pretrained ImageNet)
- **Parameters**: ~11M
- **Speed**: ~15ms per image
- **Use case**: High accuracy when data is limited

### YOLO (Future)
- **Purpose**: Real-time object detection
- **Models**: YOLOv8 Nano → Large
- **Speed**: 2-50ms per image depending on size

## Training Pipeline

```python
from perception.datasets import RobotVisionDataset
from perception.models import CNNBaseline
import torch
from torch.utils.data import DataLoader

# Create dataset
dataset = RobotVisionDataset("data/processed/images")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = CNNBaseline(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(50):
    for batch in train_loader:
        images, labels = batch["image"], batch["label"]
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Integration with MLflow

Track experiments:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"lr": 1e-3, "epochs": 50})
    
    # Training...
    
    mlflow.log_metrics({"accuracy": 0.95, "loss": 0.12})
    mlflow.pytorch.log_model(model, "model")
```

## Model Registry

Register trained models:
```bash
mlflow models create-version \
  --model-name perception-model \
  --source runs/abc123/model
```

## Deployment

### Local Inference
```python
from perception.models import CNNBaseline
import torch

model = torch.load("checkpoints/best_model.pt")
model.eval()

image = torch.randn(1, 3, 224, 224)
logits = model(image)
```

### ONNX Inference (Production)
```python
import onnxruntime as rt

sess = rt.InferenceSession("models/perception.onnx")
output = sess.run(None, {"image": image_np})
```

## Performance Targets

- **Latency**: <50ms per image (ensure real-time)
- **Accuracy**: >90% on test set
- **Memory**: <500MB model size
- **Throughput**: >20 images/sec

## Common Issues

### Out of Memory
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Use gradient accumulation
for i, batch in enumerate(loader):
    loss = model(batch)
    (loss / accum_steps).backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
```

### Poor Accuracy
- Check data augmentation settings
- Verify labels are correct
- Try learning rate schedule: `lr = 1e-3 * 0.1 ** (epoch // 10)`
- Increase model capacity or use pretrained backbone

### Slow Inference
- Use ONNX or TorchScript export
- Quantize model (int8)
- Reduce input resolution

## Next Steps

- [ ] Implement YOLO detector
- [ ] Add data augmentation pipeline
- [ ] Multi-task learning (detection + segmentation)
- [ ] Knowledge distillation for mobile
- [ ] Continuous learning from robot feedback
