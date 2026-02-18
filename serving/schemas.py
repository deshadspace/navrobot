"""
Serving schemas for API requests/responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ImageRequest(BaseModel):
    """Request with image data."""
    image_base64: str = Field(..., description="Base64 encoded image")
    metadata: Optional[dict] = None


class ObjectDetection(BaseModel):
    """Object detection result."""
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    """Response with detections."""
    model_name: str
    detections: List[ObjectDetection]
    latency_ms: float


class ActionResponse(BaseModel):
    """Response with action."""
    action: List[float] = Field(..., description="[v_linear, v_angular]")
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    images: List[str] = Field(..., description="List of base64 images")
    batch_size: int = Field(default=32, ge=1, le=256)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[dict]
    total_latency_ms: float
