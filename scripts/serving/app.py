# serving/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import torch
from ultralytics import YOLO
from stable_baselines3 import PPO
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Robot Navigation API", version="1.0.0")

# Global model holders
perception_model = None
navigation_model = None

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class NavigationAction(BaseModel):
    linear_velocity: float
    angular_velocity: float
    confidence: float

class RobotState(BaseModel):
    position: List[float]  # [x, y, theta]
    image: Optional[str] = None  # base64 encoded

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global perception_model, navigation_model
    
    try:
        # Load YOLO perception model
        perception_model = YOLO('models/obstacle_detector.pt')
        print("Perception model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load perception model: {e}")
    
    try:
        # Load RL navigation model
        navigation_model = PPO.load('models/navigation_policy.zip')
        print("Navigation model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load navigation model: {e}")

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "perception_model": perception_model is not None,
        "navigation_model": navigation_model is not None
    }

@app.post("/detect", response_model=List[DetectionResult])
async def detect_obstacles(file: UploadFile = File(...)):
    """
    Detect obstacles in uploaded image
    
    Args:
        file: Image file (jpg, png)
        
    Returns:
        List of detected objects with bounding boxes
    """
    if perception_model is None:
        raise HTTPException(status_code=503, detail="Perception model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Run detection
        results = perception_model(image)[0]
        
        # Parse results
        detections = []
        for box in results.boxes:
            detection = DetectionResult(
                class_name=results.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=box.xyxy[0].tolist()
            )
            detections.append(detection)
        
        return detections
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/navigate", response_model=NavigationAction)
async def get_navigation_action(file: UploadFile = File(...)):
    """
    Get navigation action from current observation
    
    Args:
        file: Current camera image
        
    Returns:
        Navigation action (velocities)
    """
    if navigation_model is None:
        raise HTTPException(status_code=503, detail="Navigation model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((64, 64))
        
        # Convert to numpy array
        obs = np.array(image)
        
        # Get action from policy
        action, _ = navigation_model.predict(obs, deterministic=True)
        
        return NavigationAction(
            linear_velocity=float(action[0]),
            angular_velocity=float(action[1]),
            confidence=0.95  # Placeholder
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation failed: {str(e)}")

@app.post("/inference")
async def full_inference(file: UploadFile = File(...)):
    """
    Run full perception + navigation pipeline
    
    Returns both detections and navigation action
    """
    if perception_model is None or navigation_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Detection
        detect_results = perception_model(image)[0]
        detections = []
        for box in detect_results.boxes:
            detections.append({
                'class': detect_results.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        
        # Navigation
        image_resized = image.resize((64, 64))
        obs = np.array(image_resized)
        action, _ = navigation_model.predict(obs, deterministic=True)
        
        return {
            'detections': detections,
            'navigation': {
                'linear_velocity': float(action[0]),
                'angular_velocity': float(action[1])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
