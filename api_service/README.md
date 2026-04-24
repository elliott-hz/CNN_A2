# Dog Emotion Recognition API Service

FastAPI backend for dog face detection and emotion classification.

## Quick Start

### 1. Install Dependencies

```bash
# First, install PyTorch (choose based on your hardware)

# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt

# Ensure NumPy compatibility
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

### 2. Verify Model Files

Make sure you have the trained models in `../best_models/`:
- `detection_YOLOv8_baseline.pt`
- `emotion_ResNet50_baseline.pth`

### 3. Run the Server

```bash
python main.py
```

The API will start on `http://localhost:8000`

### 4. Test the API

Open your browser and visit:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST /api/detect

Upload an image to detect dog faces and classify emotions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "dog_id": 0,
      "bbox": [100.5, 150.2, 300.8, 400.6],
      "detection_confidence": 0.95,
      "emotion": "happy",
      "emotion_confidence": 0.87,
      "emotion_probabilities": {
        "angry": 0.02,
        "happy": 0.87,
        "relaxed": 0.05,
        "frown": 0.03,
        "alert": 0.03
      }
    }
  ],
  "message": "Detected 1 dog(s)"
}
```

## Architecture

```
Client (React Frontend)
       ↓ HTTP POST /api/detect
FastAPI Server
       ↓ Python inference
PipelineInference (YOLOv8 + ResNet50)
       ↓ Results
JSON Response with bbox + emotion
```

## Development

- **CORS**: Configured to allow requests from `http://localhost:5173` (Vite default)
- **Model Loading**: Models are loaded once at startup for better performance
- **Device**: Automatically detects and uses GPU if available, otherwise CPU
