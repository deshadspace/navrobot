"""
Serving utilities and helpers.

Common functions for the API service.
"""

import base64
import numpy as np
from typing import Tuple
from io import BytesIO


def encode_image_base64(image: np.ndarray) -> str:
    """
    Encode numpy array to base64 string.
    
    Args:
        image: image array
        
    Returns:
        base64 encoded string
    """
    from PIL import Image
    
    img = Image.fromarray((image * 255).astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()


def decode_image_base64(image_b64: str) -> np.ndarray:
    """
    Decode base64 string to numpy array.
    
    Args:
        image_b64: base64 encoded image
        
    Returns:
        image array
    """
    from PIL import Image
    
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data))
    
    return np.array(image).astype(np.float32) / 255.0


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: input image
        target_size: target dimensions
        
    Returns:
        preprocessed image
    """
    from PIL import Image
    
    # Resize
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    pil_image = pil_image.resize(target_size)
    resized = np.array(pil_image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized = (resized - mean) / std
    
    return normalized


def format_response(predictions: np.ndarray, model_name: str) -> dict:
    """
    Format predictions as response.
    
    Args:
        predictions: model predictions
        model_name: name of model
        
    Returns:
        response dict
    """
    return {
        "model": model_name,
        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        "timestamp": str(np.datetime64('now')),
    }
