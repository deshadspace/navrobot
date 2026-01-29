# perception/train.py
import os
import torch
from ultralytics import YOLO
import yaml

def train_obstacle_detector(
    data_yaml_path='configs/data.yaml',
    epochs=50,
    batch_size=16,
    img_size=640,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train YOLO model for obstacle detection
    
    Args:
        data_yaml_path: Path to data configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Training device
    """
    print(f"Training on device: {device}")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='experiments/runs/detect',
        name='obstacle_detection',
        save=True,
        verbose=True
    )
    
    print("Training completed!")
    return results

def evaluate_model(model_path, data_yaml_path):
    """Evaluate trained model"""
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(data=data_yaml_path)
    
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def export_model(model_path, format='onnx'):
    """Export model for deployment"""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Model exported to {format}")

if __name__ == '__main__':
    # Create data config if it doesn't exist
    data_config = {
        'path': 'data/processed',
        'train': 'images',
        'val': 'images',
        'nc': 2,  # number of classes
        'names': ['obstacle', 'human']
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    # Train
    results = train_obstacle_detector(
        data_yaml_path='configs/data.yaml',
        epochs=10,  # Start small for testing
        batch_size=8
    )
