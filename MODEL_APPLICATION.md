# Model Application & Web Interface Guide

This document provides comprehensive information about the web application architecture, API documentation, inference pipeline, and deployment guide for the Visual Dog Emotion Recognition system.

---

## 🎯 Web Application Overview

### Features

The web application provides a user-friendly interface for real-time dog emotion recognition with **three interaction modes**:

#### 📷 Mode 1: Upload Image
- ✅ **Image Upload**: Drag & drop or click to upload images
- ✅ **Instant Analysis**: Automatic detection upon upload
- ✅ **Visual Annotations**: Bounding boxes drawn directly on uploaded images
- ✅ **Detailed Results**: Confidence scores and emotion probabilities

#### 🎬 Mode 2: Upload Video (Enhanced in v3.1.0)
- ✅ **Video Upload**: Support for MP4, WebM, AVI files (max 20 seconds, 50MB)
- ✅ **Video Playback**: Native HTML5 video player with controls
- ✅ **Smooth Video Annotations**: Bounding boxes smoothly follow dog movement using linear interpolation
- ✅ **Pre-processing Analysis**: Backend analyzes all frames upfront at 5fps (every 200ms)
- ✅ **Fluent Animation**: Real-time boundary box interpolation for buttery-smooth tracking
- ✅ **Real-Time Progress Updates**: Live progress bar showing frame-by-frame processing status via Server-Sent Events (SSE)
- ✅ **Optimized Performance**: Direct memory processing eliminates file I/O overhead

#### 📹 Mode 3: Live Stream
- ✅ **Camera Access**: Real-time webcam feed using getUserMedia API
- ✅ **Live Indicator**: Visual feedback showing active stream
- ✅ **Future Ready**: Framework for real-time emotion detection

### Supported Emotions

- 😊 **Happy**: Joyful, playful expression
- 😠 **Angry**: Aggressive, threatening posture
- 😌 **Relaxed**: Calm, peaceful state
- 😟 **Frown**: Sad, concerned look
- 👀 **Alert**: Attentive, watchful stance

---

## 🏗️ Web App Architecture

### Tech Stack

**Frontend:**
- React 18.x
- Vite (build tool with hot reload)
- Axios (HTTP client)
- CSS Modules (styling)

**Backend:**
- FastAPI (async web framework)
- PyTorch (deep learning)
- Ultralytics YOLOv8 (detection)
- OpenCV & PIL (image processing)

### Architecture Diagram

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────────┐
│  React Frontend │  HTTP   │ FastAPI      │  Python │ Model Pipeline  │
│  (Vite + Axios) │ ◄─────► │ Backend      │ ◄─────► │ (YOLO+ResNet)   │
│  localhost:5173 │  JSON   │ localhost:8000│        │                 │
└─────────────────┘         └──────────────┘         └─────────────────┘
```

### Directory Structure

```
CNN_A3/
├── api_service/              # Backend API service
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # API documentation
│
├── web_intf/                # Frontend React app
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ImageUploader.jsx    # Upload & preview
│   │   │   ├── ResultsDisplay.jsx   # Results with canvas
│   │   │   ├── VideoUploader.jsx    # Video upload interface
│   │   │   ├── VideoResultsDisplay.jsx # Video playback & annotations
│   │   │   ├── LiveStream.jsx       # Live camera stream
│   │   │   └── *.css                # Component styles
│   │   ├── services/
│   │   │   └── api.js       # API client
│   │   ├── App.jsx          # Main app component
│   │   └── App.css          # Global styles
│   ├── package.json
│   └── vite.config.js
│
├── best_models/             # Trained models (shared)
│   ├── detection_YOLOv8_baseline.pt
│   └── emotion_ResNet50_baseline.pth
│
├── src/                     # Existing ML code (reused)
│   └── inference/
│       └── pipeline_inference.py
│
├── start_web_app.sh         # One-command startup script
└── test_web_app.py          # Automated test suite
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- PyTorch (CPU or GPU version)
- Trained model files in `best_models/` directory

### Installation

#### 1. Install Backend Dependencies

```bash
cd api_service

# First, install PyTorch (choose based on your hardware)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Ensure NumPy compatibility
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

#### 2. Install Frontend Dependencies

```bash
cd web_intf
npm install
```

### Running the Application

#### Option A: One-Command Start (Recommended)

```bash
chmod +x start_web_app.sh
./start_web_app.sh
```

This script will automatically:
- Check model files exist
- Start backend API on port 8000
- Start frontend dev server on port 5173
- Handle cleanup when you press Ctrl+C

#### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd api_service
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd web_intf
npm run dev
```

### Access Points

Once started, open these URLs in your browser:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend App** | http://localhost:5173 | Main user interface |
| **Backend API** | http://localhost:8000 | API root endpoint |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Health Check** | http://localhost:8000/health | API status check |

### Stopping the Application

**If using start_web_app.sh:**
Press `Ctrl+C` in the terminal

**If running manually:**
```bash
# Stop backend
pkill -f "python main.py"

# Stop frontend
pkill -f "npm run dev"
```

---

## 📖 Using the Web Application

### Step-by-Step Guide

#### 📷 Mode 1: Upload Image Mode

1. **Select Mode**: Click "📷 Upload Image" button in header (default mode)
2. **Upload an image**: 
   - Click the upload area, OR
   - Drag & drop an image file
   - **Detection starts automatically** - no button click needed!
3. **View results**: See annotated image with:
   - Colored bounding boxes around detected dogs
   - Emotion labels at top-left of each box
   - Dog ID tags at bottom-left
   - Detailed metrics cards below
4. **Analyze another image**: Simply click the upload area again or drag & drop a new image (previous results auto-clear)

#### 🎬 Mode 2: Upload Video Mode

1. **Select Mode**: Click "🎬 Upload Video" button in header
2. **Upload a video**:
   - Click the upload area, OR
   - Drag & drop a video file
   - Supported formats: MP4, WebM, AVI (max 50MB, 20 seconds)
3. **Automatic Preprocessing**:
   - System extracts frames every 200ms (5fps)
   - Backend processes all frames in batch
   - Progress indicator shows processing status
4. **Video Playback**: 
   - Video loads and plays automatically after analysis
   - Use play/pause controls to manage playback
   - Smooth bounding box animations follow dog movement
5. **Switch Videos**: Click "🔄 Change Video" to upload a different video

#### 📹 Mode 3: Live Stream Mode

1. **Select Mode**: Click "📹 Live Stream" button in header
2. **Grant Camera Permission**: Browser will request camera access - click "Allow"
3. **View Live Feed**: 
   - Real-time camera feed displays
   - "LIVE" indicator shows stream is active
   - Future enhancement: Real-time emotion detection overlay

### Supported File Formats

**Images:**
- JPEG/JPG
- PNG
- Maximum size: 10MB
- Auto-resized to 640px max dimension for optimal performance

**Videos:**
- MP4 (recommended)
- WebM
- AVI
- Maximum size: 50MB
- Maximum duration: 20 seconds

### Visual Annotations

When results are displayed, you'll see:

**On Images/Video Frames:**
- **Colored Bounding Boxes**: Each emotion has a unique color
  - 😊 Happy: Green (#4CAF50)
  - 😠 Angry: Red (#f44336)
  - 😌 Relaxed: Blue (#2196F3)
  - 😟 Frown: Orange (#FF9800)
  - 👀 Alert: Purple (#9C27B0)

- **Emotion Labels**: Smart positioning at top-left of each box
  - Shows emoji + emotion name + confidence %
  - Example: "😊 Happy (87.3%)"
  - **Auto-adjusts position**: If box is near image edge, label moves inside to stay visible

- **Dog ID Tags**: Smart positioning at bottom-left of each box
  - Shows "Dog #1", "Dog #2", etc.
  - **Auto-adjusts position**: If box is near bottom edge, label moves inside to stay visible

### UI Optimization

**Three-Mode Interface**:
- Clear mode buttons in header with active state highlighting
- Smooth transitions between modes
- Independent state management for each mode
- Automatic cleanup when switching modes

**Compact Upload Interface**:
- Upload area uses minimal vertical space (80px height)
- Horizontal layout with icon + text side-by-side
- File info shown inline (name + size)
- No duplicate rendering

**Frontend Image Resizing**:
- Images >640px automatically resized to 640px max dimension
- Aspect ratio preserved
- 90% JPEG quality for optimal size/quality balance
- Reduces inference time by ~60% on CPU

**Smart Label Positioning**:
- Labels automatically repositioned when near image edges
- Top label (emotion): Moves inside box if <30px space above
- Bottom label (dog ID): Moves inside box if near canvas bottom
- Ensures 100% label visibility in all scenarios

---

## 🔌 API Documentation

### POST /api/detect

Upload an image to detect dog faces and classify emotions.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog_image.jpg"
```

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

### POST /api/detect-base64

Detect emotions from base64-encoded image (used for video frames).

**Request:**
```javascript
fetch('http://localhost:8000/api/detect-base64', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_base64: 'data:image/jpeg;base64,/9j/4AAQSkZJRg...'
  })
})
```

**Response:** Same format as `/api/detect`

### POST /api/analyze-video-batch

Batch process video frames for maximum speed (NEW in v3.2.0).

**Request:**
```bash
curl -X POST "http://localhost:8000/api/analyze-video-batch" \
  -F "file=@video.mp4" \
  -F "batch_size=10"
```

**Response:**
```json
{
  "success": true,
  "frames": [
    {
      "timestamp": 0.0,
      "results": [...]  // Same format as /api/detect
    },
    ...
  ],
  "total_frames": 100,
  "processing_time_seconds": 5.2
}
```

### POST /api/analyze-video-stream

Stream video analysis with real-time progress updates.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/analyze-video-stream" \
  -F "file=@video.mp4"
```

**Response:** Server-Sent Events (SSE) stream with progress updates:
```
data: {"progress": 10, "current_frame": 10, "total_frames": 100}
data: {"progress": 20, "current_frame": 20, "total_frames": 100}
...
data: {"complete": true, "frames": [...]}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "CPU"
}
```

### Interactive API Docs

Visit http://localhost:8000/docs for Swagger UI with:
- All available endpoints
- Request/response schemas
- Try-it-out functionality
- Authentication options (if added later)

---

## ⚡ Performance & Optimization

### Performance Metrics

| Metric | CPU (Mac M1) | GPU (T4) |
|--------|--------------|----------|
| Single Image Inference | ~500ms | ~100ms |
| Memory Usage (Backend) | ~2GB | ~4GB |
| Memory Usage (Frontend) | ~100MB | ~100MB |
| Max Concurrent Users | ~5 | ~50 |

### Video Processing Performance

| Method | 2MB Video (50 frames) | Speed | Progress Feedback | Best For |
|--------|----------------------|-------|-------------------|----------|
| **Sequential + File I/O** (Legacy) | ~15-18s | 1x | ❌ None | Legacy systems |
| **Sequential + Memory** (v3.1.0) | ~10-12s | 1.5x | ✅ SSE Streaming | Large videos, real-time feedback |
| **Batch Processing** (v3.2.0 NEW) | **~5-7s** | **3x** | ⚡ Simulated | **Small videos (<5MB), maximum speed** |

### Configuration Options

#### Backend Configuration

Edit `api_service/main.py`:

```python
# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    ...
)

# Inference parameters
results = pipeline.predict(temp_path, conf=0.5, iou=0.45)
# conf: Detection confidence threshold (0.0-1.0)
# iou: NMS IoU threshold (0.0-1.0)
```

#### Frontend Configuration

Edit `web_intf/src/services/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Backend URL
```

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: Models not loading
```
Solution: Verify model files exist in best_models/
- detection_YOLOv8_baseline.pt (~50 MB)
- emotion_ResNet50_baseline.pth (~98 MB)
```

**Problem**: Port 8000 already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill
```

**Problem**: Import errors
```bash
cd api_service
pip install -r requirements.txt
```

### Frontend Issues

**Problem**: Cannot connect to API
```
Solution: 
1. Check if backend is running: curl http://localhost:8000/health
2. Verify CORS settings in api_service/main.py
3. Check browser console for error messages
```

**Problem**: npm install fails
```bash
# Clear npm cache
npm cache clean --force
# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Performance Issues

**Problem**: Slow inference on CPU

Expected inference time on CPU: ~500ms - 1s per image

Solutions:
1. Use smaller images (< 2MB recommended)
2. Close other applications to free CPU resources
3. Consider using GPU for production deployment

### Testing

Run the automated test suite:

```bash
python test_web_app.py
```

This checks:
- Model files existence
- Backend API health
- Frontend accessibility
- API documentation availability

---

## 🚀 Deployment Guide

### Production Deployment Recommendations

For production deployment, consider:

1. **GPU Server**: Deploy on GPU-enabled instance for better performance
2. **Docker**: Containerize both services for easy deployment
3. **Database**: Add PostgreSQL for result history
4. **Authentication**: Implement JWT-based user auth
5. **Rate Limiting**: Prevent API abuse
6. **Logging**: Structured logging with ELK stack
7. **Monitoring**: Prometheus + Grafana dashboards
8. **CDN**: Serve static assets via CDN
9. **HTTPS**: SSL/TLS certificates
10. **Load Balancing**: Nginx reverse proxy

### Docker Deployment (Example)

**Backend Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY api_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:16-alpine AS build
WORKDIR /app
COPY web_intf/package*.json ./
RUN npm install
COPY web_intf/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Variables

Create `.env` file for production:

```bash
# Backend
MODEL_DETECTION_PATH=/models/detection_YOLOv8_baseline.pt
MODEL_CLASSIFICATION_PATH=/models/emotion_ResNet50_baseline.pth
DEVICE=cuda
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Frontend
REACT_APP_API_URL=https://api.yourdomain.com
```

---

## 🔮 Future Enhancements

- [x] ~~Real-time webcam support~~ ✅ **COMPLETED in v2.0.0** - Live camera with frame capture
- [x] ~~Video upload and analysis~~ ✅ **COMPLETED in v3.0.0** - Video file upload with periodic frame processing
- [x] ~~Smooth video annotations~~ ✅ **COMPLETED in v3.1.0** - Linear interpolation for smooth bbox tracking
- [x] ~~Batch video processing~~ ✅ **COMPLETED in v3.2.0** - Parallel batch inference for faster processing
- [ ] Real-time WebSocket streaming for sub-second latency
- [ ] Batch processing for multiple images
- [ ] Save detection history to database
- [ ] User authentication and accounts
- [ ] Export results as CSV/JSON
- [ ] Mobile-responsive improvements
- [ ] Model performance monitoring
- [ ] Docker containerization

---

## 📚 Related Documentation

- **Data Preprocessing**: See [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) for dataset preparation
- **Model Training**: See [MODEL_TRAINING.md](MODEL_TRAINING.md) for experiment configurations
- **Project Overview**: See [README.md](README.md) for architecture summary
