"""
Dog Emotion Recognition API Service
FastAPI backend for dog face detection and emotion classification
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.pipeline_inference import PipelineInference


# Pydantic models for API response
class DetectionResult(BaseModel):
    dog_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    detection_confidence: float
    emotion: str
    emotion_confidence: float
    emotion_probabilities: Dict[str, float]


class InferenceResponse(BaseModel):
    success: bool
    results: List[DetectionResult]
    message: str = ""


# Initialize FastAPI app
app = FastAPI(
    title="Dog Emotion Recognition API",
    description="API for dog face detection and emotion classification",
    version="1.0.0"
)

# CORS middleware - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global pipeline instance (loaded once at startup)
pipeline = None


@app.on_event("startup")
async def load_models():
    """Load models when the application starts"""
    global pipeline
    
    print("=" * 80)
    print("Loading Dog Emotion Recognition Models...")
    print("=" * 80)
    
    try:
        # Model paths
        detection_model_path = Path(__file__).parent.parent / "best_models" / "detection_YOLOv8_baseline.pt"
        classification_model_path = Path(__file__).parent.parent / "best_models" / "emotion_ResNet50_baseline.pth"
        
        # Verify model files exist
        if not detection_model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {detection_model_path}")
        if not classification_model_path.exists():
            raise FileNotFoundError(f"Classification model not found: {classification_model_path}")
        
        # Initialize pipeline
        pipeline = PipelineInference(
            detection_model_path=str(detection_model_path),
            classification_model_path=str(classification_model_path)
        )
        
        # Check device
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"\n✅ Models loaded successfully!")
        print(f"🖥️  Running on: {device}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dog Emotion Recognition API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "models_loaded": pipeline is not None,
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    }


@app.post("/api/detect", response_model=InferenceResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """
    Detect dog faces and classify emotions in uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        List of detections with bounding boxes and emotion labels
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        contents = await file.read()
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to temporary file for pipeline (YOLOv8 expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, format='JPEG')
            temp_path = tmp.name
        
        try:
            # Run inference
            results = pipeline.predict(temp_path, conf=0.5, iou=0.45)
            
            # Format response
            detection_results = []
            for result in results:
                detection_results.append(DetectionResult(
                    dog_id=result['dog_id'],
                    bbox=result['bbox'],
                    detection_confidence=result['detection_confidence'],
                    emotion=result['emotion'],
                    emotion_confidence=result['emotion_confidence'],
                    emotion_probabilities=result['emotion_probabilities']
                ))
            
            message = f"Detected {len(detection_results)} dog(s)" if detection_results else "No dogs detected"
            
            return InferenceResponse(
                success=True,
                results=detection_results,
                message=message
            )
        
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
