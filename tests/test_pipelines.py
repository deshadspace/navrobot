"""
Test suite for pipelines.
"""

import pytest
from pathlib import Path
from pipelines.data_pipeline import DataPipeline
from pipelines.training_pipeline import TrainingPipeline


def test_data_pipeline_creation():
    """Test pipeline initialization."""
    pipeline = DataPipeline()
    
    assert pipeline is not None
    assert pipeline.raw_path is not None
    assert pipeline.processed_path is not None


def test_training_pipeline_creation():
    """Test training pipeline initialization."""
    pipeline = TrainingPipeline()
    
    assert pipeline is not None
    assert pipeline.config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
