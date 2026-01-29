# perception/datasets/vision_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ObstacleDetectionDataset(Dataset):
    """Dataset for obstacle detection using YOLO format"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory with 'images' and 'labels' subdirectories
            transform: Optional transform to be applied to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        sample = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class VisionTransform:
    """Transform for vision dataset"""
    
    def __init__(self, image_size=640):
        self.image_size = image_size
    
    def __call__(self, sample):
        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        
        # Resize image
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        sample['image'] = image
        sample['boxes'] = torch.from_numpy(boxes)
        sample['labels'] = torch.from_numpy(labels)
        
        return sample
