"""
Test suite for perception module.
"""

import pytest
import torch
import numpy as np
from perception.models.cnn_baseline import CNNBaseline
from perception.datasets.vision_dataset import RobotVisionDataset


def test_cnn_baseline():
    """Test CNN model creation and forward pass."""
    model = CNNBaseline(num_classes=10)
    
    # Dummy input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 10)


def test_cnn_feature_extraction():
    """Test feature extraction."""
    model = CNNBaseline(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    
    features = model.extract_features(x)
    
    assert features.shape[0] == 2  # batch size
    assert len(features.shape) == 2


def test_dataset():
    """Test dataset creation."""
    import tempfile
    from pathlib import Path
    
    # Create temporary directory with dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy PNG files
        for i in range(10):
            np.random.rand(64, 64, 3)
            # In real test, would save as PNG
        
        # Note: This is a simplified test
        # In production, would save actual PNG files


def test_model_training_step():
    """Test training loop."""
    model = CNNBaseline(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy batch
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 10, (4,))
    
    # Forward pass
    logits = model(x)
    loss = criterion(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
