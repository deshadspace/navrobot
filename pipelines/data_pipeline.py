"""
Data preprocessing pipeline.

End-to-end data cleaning, validation, and preparation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class DataPipeline:
    """ETL pipeline for robot data."""
    
    def __init__(self, config_path: Path = Path("configs/training.yaml")):
        """
        Args:
            config_path: path to training config
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config["data"]["raw_data_path"])
        self.processed_path = Path(self.config["data"]["processed_data_path"])
    
    def extract(self) -> List[Dict]:
        """Extract raw data from files."""
        episodes = []
        
        for episode_file in sorted(self.raw_path.glob("*.npz")):
            data = np.load(episode_file)
            episodes.append({
                "images": data["images"],
                "states": data["states"],
                "actions": data["actions"],
                "rewards": data["rewards"],
            })
        
        print(f"✓ Extracted {len(episodes)} episodes")
        return episodes
    
    def transform(self, episodes: List[Dict]) -> Tuple[List, List]:
        """Transform and augment data."""
        processed_images = []
        processed_labels = []
        
        for episode in episodes:
            images = episode["images"]
            states = episode["states"]
            
            # Resize images
            resized = self._resize_images(images)
            processed_images.extend(resized)
            
            # Normalize states
            normalized = self._normalize_states(states)
            processed_labels.extend(normalized)
        
        print(f"✓ Transformed {len(processed_images)} samples")
        return processed_images, processed_labels
    
    def load(self, images: List, labels: List) -> None:
        """Save processed data."""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save images
        np.save(self.processed_path / "images.npy", np.array(images))
        
        # Save labels
        np.save(self.processed_path / "labels.npy", np.array(labels))
        
        print(f"✓ Saved processed data to {self.processed_path}")
    
    def _resize_images(self, images: np.ndarray) -> List[np.ndarray]:
        """Resize images to target size."""
        # Placeholder: would use cv2.resize
        return [np.expand_dims(img, 0) for img in images]
    
    def _normalize_states(self, states: np.ndarray) -> List[np.ndarray]:
        """Normalize state vectors."""
        # Standardization
        mean = states.mean(axis=0)
        std = states.std(axis=0) + 1e-8
        normalized = (states - mean) / std
        return list(normalized)
    
    def run(self) -> None:
        """Execute full pipeline."""
        episodes = self.extract()
        images, labels = self.transform(episodes)
        self.load(images, labels)
        print("✓ Data pipeline complete")


def create_splits(
    data_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
):
    """Create train/val/test splits."""
    images = np.load(data_dir / "images.npy")
    labels = np.load(data_dir / "labels.npy")
    
    n = len(images)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    indices = np.random.permutation(n)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        "train": (images[train_indices], labels[train_indices]),
        "val": (images[val_indices], labels[val_indices]),
        "test": (images[test_indices], labels[test_indices]),
    }
