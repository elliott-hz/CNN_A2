# Visual Dog Emotion Recognition - Full CNN Pipeline Design
*Fully aligned with Assignment 3 PartB | 2 datasets • 2 models • 1 end-to-end pipeline*

---

## 1. Overall Architecture Overview
This project uses a **two-stage deep learning pipeline** (standard for object detection + image classification tasks).

### Core Logic
- **Stage 1**: Dog face detection (localize where the dog face is)
- **Stage 2**: Dog emotion classification (classify the emotion of the cropped face)
- **Two datasets** for two different tasks
- **Two models** trained separately
- **One unified pipeline** for inference

```
Input Image / Video Frame
          ↓
┌─────────────────────────┐
│   Stage 1: Dog Face Detection   │
│   Model: YOLOv8 / Faster R-CNN           │
│   Trained on Detection Dataset  │
└─────────────────────────┘
          ↓ (crop face region)
┌─────────────────────────┐
│   Stage 2: Emotion Classification │
│   Model: ResNet50 (CNN)          │
│   Trained on Emotion Dataset     │
└─────────────────────────┘
          ↓
Output: Bounding Box + Emotion Label (happy/angry/relaxed/frown/alert)
```

---

## 2. Dataset Design (2 Datasets – Fully Reasonable & Standard)
### 2.1 Dataset 1: Dog Face Detection
- **Purpose**: Train model to **localize dog faces** (bounding box)
- **Source**: Dog Face Detection Dataset (Kaggle)
- **Size**: ~6,000 images
- **Annotation**: Bounding boxes (x1, y1, x2, y2)
- **Task type**: Object detection
- **Train/Val/Test split**: 70% / 20% / 10%

### 2.2 Dataset 2: Dog Emotion Classification
- **Purpose**: Train model to **classify 5 emotion categories**
- **Source**: Dog Emotions – 5 Classes (Kaggle)
- **Size**: ~9,000 images (~1,800 per class)
- **5 Classes**: angry, happy, relaxed, frown, alert
- **Task type**: Image classification
- **Train/Val/Test split**: 70% / 20% / 10%

---

## 3. Model Design (2 Models – Trained Separately)
### 3.1 Model 1: Dog Face Detector – YOLOv8 (Recommended) or Faster R-CNN
- **Backbone**: CSPDarknet (YOLOv8) or ResNet50 (Faster R-CNN)
- **Task**: Output bounding box for dog faces
- **Key outputs**:
  - Bounding box coordinates
  - Confidence score
- **Loss function**:
  - CIoU loss (YOLOv8) or Smooth L1 + Cross-entropy (Faster R-CNN)
- **Evaluation metrics**: mAP@0.5, mAP@0.5:0.95, IoU
- **Model Selection Rationale**:
  - **YOLOv8**: Faster inference (~30-50 FPS on T4), easier deployment, suitable for real-time applications
  - **Faster R-CNN**: Higher accuracy for small/occluded faces, but slower (~5-10 FPS)
  - **Decision**: Start with YOLOv8 for efficiency; switch to Faster R-CNN if accuracy is insufficient

### 3.2 Model 2: Emotion Classifier – ResNet50 (CNN)
- **Type**: Deep convolutional neural network
- **Task**: 5-class emotion classification
- **Input**: Cropped and preprocessed dog face image (224×224)
- **Output**: Probability over 5 emotions
- **Loss function**: Cross-entropy loss (with class weighting if needed)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 4. Training Pipeline (Separate Training for 2 Models)
### 4.1 Hardware & Environment Setup
- **Primary Training Environment**: AWS SageMaker JupyterLab with NVIDIA T4 GPU (16GB VRAM)
- **Local Development**: Laptop CPU/GPU for initial code testing and debugging
- **Framework**: PyTorch 2.0+ with torchvision
- **Memory Optimization Strategies**:
  - Use mixed precision training (AMP) to reduce memory usage
  - Gradient accumulation for larger effective batch sizes
  - Reduce batch size if OOM errors occur

### 4.2 Training Phase for Model 1 – Dog Face Detection (YOLOv8/Faster R-CNN)
1. **Dataset Preparation**:
   - Load detection dataset and parse annotations (YOLO format or COCO format)
   - Verify annotation quality (check for invalid boxes)

2. **Image Preprocessing & Augmentation**:
   - Resize to model input size (640×640 for YOLOv8, variable for Faster R-CNN)
   - Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - **Data Augmentation** (critical for limited dataset):
     - Mosaic augmentation (YOLOv8 default, combines 4 images)
     - Random horizontal flip (p=0.5)
     - Random scaling (±20%)
     - Color jittering (brightness, contrast, saturation ±30%)
     - MixUp augmentation (blend two images)
     - CutOut/CutMix (randomly mask regions)

3. **Model Initialization**:
   - Load pretrained weights (COCO dataset for transfer learning)
   - Freeze backbone layers initially, unfreeze gradually

4. **Training Configuration**:
   - **Optimizer**: SGD with momentum (0.9) or AdamW
   - **Learning Rate**: 
     - Initial LR: 1e-3 with warmup (first 5 epochs)
     - Scheduler: Cosine annealing or step decay (reduce by 0.1 every 20 epochs)
   - **Batch Size**: 8-16 (depending on GPU memory, use gradient accumulation if needed)
   - **Epochs**: 50-100 (with early stopping patience=10)
   - **Weight Decay**: 1e-4 (L2 regularization)
   - **Mixed Precision**: Enable AMP (Automatic Mixed Precision) for faster training

5. **Monitoring & Validation**:
   - Track training/validation loss per epoch
   - Evaluate mAP on validation set every 5 epochs
   - Save best model based on validation mAP

6. **Save trained detection model** (.pt or .pth format)

### 4.3 Training Phase for Model 2 – ResNet50 (Emotion Classification)
1. **Dataset Preparation**:
   - Load emotion classification dataset with folder structure
   - Verify class balance (≈1,800 images per class)

2. **Image Preprocessing & Augmentation**:
   - Resize to 224×224 pixels
   - Normalize to ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - **Data Augmentation** (to prevent overfitting on small dataset):
     - Random rotation (±15 degrees)
     - Random horizontal flip (p=0.5)
     - Random vertical flip (p=0.2, optional)
     - Brightness adjustment (±20%)
     - Contrast adjustment (±20%)
     - Gaussian noise injection (σ=0.01-0.05)
     - Random erasing (mask random patches)
     - RandAugment or AutoAugment (advanced augmentation policies)

3. **Model Setup**:
   - Load pretrained ResNet50 (ImageNet weights)
   - Replace final fully connected layer: `nn.Linear(2048, 5)` 
   - Add dropout layer (p=0.5) before final FC layer
   - Freeze backbone layers initially (first 10-20 epochs), then unfreeze for fine-tuning

4. **Training Configuration**:
   - **Optimizer**: Adam (lr=1e-4, betas=(0.9, 0.999)) or SGD with momentum
   - **Learning Rate**: 
     - Phase 1 (frozen backbone): lr=1e-3 for 10-20 epochs
     - Phase 2 (fine-tuning): lr=1e-4 to 1e-5 for 20-30 epochs
     - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
   - **Batch Size**: 32-64 (adjust based on GPU memory)
   - **Epochs**: 30-50 total (with early stopping patience=7)
   - **Loss Function**: Cross-entropy loss with label smoothing (ε=0.1)
   - **Class Weighting**: Calculate inverse frequency weights if class imbalance detected
   - **Regularization**:
     - Dropout (p=0.5 in final layers)
     - Batch normalization (already in ResNet)
     - Weight decay: 1e-4
   - **Mixed Precision**: Enable AMP for memory efficiency

5. **Monitoring & Validation**:
   - Track training/validation loss and accuracy per epoch
   - Compute confusion matrix on validation set
   - Monitor per-class precision/recall to detect weak classes
   - Save best model based on validation accuracy

6. **Save trained classification model** (.pth format)

---

## 5. Inference Pipeline (Unified – 1 Pipeline)
### 5.1 Standard Inference Flow
```
1. Read input (image / video frame)
2. Preprocess input for detection model (resize, normalize)
3. Run Model 1 (YOLOv8/Faster R-CNN) → detect dog face regions
4. Filter low-confidence boxes (confidence threshold: 0.5-0.7)
5. Apply Non-Maximum Suppression (NMS) to remove duplicate boxes (IoU threshold: 0.45)
6. Crop each detected dog face region with padding (10-20% margin)
7. Preprocess cropped face for classification model (resize to 224×224, normalize)
8. Run Model 2 (ResNet50 CNN) → predict emotion probabilities
9. Select highest probability class as final prediction
10. Draw bounding box and emotion label on original image
11. Display/save result
```

### 5.2 Optimized Inference Strategies
**For Real-Time Performance (Video/Live Stream)**:
1. **Batch Processing**:
   - If multiple dogs detected, batch all cropped faces for single classification forward pass
   - Reduces overhead from repeated model loading

2. **Frame Skipping**:
   - For video streams, run detection every 2-3 frames
   - Interpolate bounding boxes for skipped frames
   - Reduces computational load by 50-66%

3. **Confidence Threshold Tuning**:
   - Higher threshold (0.7) for fewer false positives
   - Lower threshold (0.4) for higher recall (more detections)
   - Adjust based on use case requirements

4. **Region-of-Interest Tracking**:
   - Track detected faces across frames using simple tracking (e.g., centroid tracking)
   - Only re-run detection when tracking confidence drops

5. **Model Optimization**:
   - Export models to ONNX format for faster inference
   - Consider INT8 quantization (if accuracy drop < 2%)
   - Use TorchScript for model compilation

**For Single Image Inference**:
- No optimization needed, prioritize accuracy
- Can afford lower confidence thresholds for better recall

### 5.3 Expected Performance Metrics
- **Detection Speed**: 
  - YOLOv8: ~30-50 FPS on T4 GPU
  - Faster R-CNN: ~5-10 FPS on T4 GPU
- **Classification Speed**: ~100-200 FPS on T4 GPU (ResNet50)
- **End-to-End Latency**: 
  - YOLOv8 pipeline: ~50-80ms per image
  - Faster R-CNN pipeline: ~150-250ms per image

---

## 6. Evaluation Plan
### 6.1 Model-Level Evaluation
**Detection Model**:
- mAP@0.5 (primary metric)
- mAP@0.5:0.95 (strict metric)
- IoU distribution analysis
- Precision-Recall curve
- Test scenarios:
  - No dog present (false positive rate)
  - Single dog (baseline performance)
  - Multiple dogs (detection completeness)
  - Partially occluded faces (robustness)
  - Small faces in large images (scale robustness)

**Classification Model**:
- Overall accuracy
- Per-class Precision, Recall, F1-score
- Confusion matrix (identify commonly confused pairs)
- Test scenarios:
  - Clear facial expressions (easy cases)
  - Ambiguous expressions (hard cases)
  - Different breeds (generalization)
  - Various lighting conditions

### 6.2 System-Level Evaluation
- End-to-end accuracy (correct detection + correct classification)
- Processing speed (FPS for video, latency for images)
- Stability on continuous video stream (no crashes/memory leaks)
- Robustness to input variations (resolution, aspect ratio, quality)

---

## 7. Error Handling & Optimization
### 7.1 Training Phase Issues

**Overfitting**:
- Symptoms: Training accuracy >> Validation accuracy
- Solutions:
  - Increase data augmentation intensity
  - Add dropout layers (increase p from 0.3 to 0.5)
  - Enable early stopping
  - Reduce model complexity (use smaller backbone)
  - Add weight decay (increase from 1e-4 to 1e-3)

**Underfitting**:
- Symptoms: Both training and validation accuracy are low
- Solutions:
  - Increase model capacity (deeper backbone)
  - Train for more epochs
  - Reduce regularization strength
  - Check learning rate (may be too high or too low)
  - Verify data preprocessing correctness

**Class Imbalance** (Classification):
- Symptoms: Some classes have much lower recall
- Solutions:
  - Use weighted cross-entropy loss (weight = 1/class_frequency)
  - Oversample minority classes during training
  - Use focal loss to focus on hard examples
  - Collect more data for underrepresented classes

**Detection Failures**:
- Symptoms: Low mAP, many missed detections
- Solutions:
  - Adjust confidence threshold (lower for higher recall)
  - Tune NMS IoU threshold
  - Improve data augmentation (especially for edge cases)
  - Check annotation quality (incorrect boxes hurt training)
  - Try different anchor box sizes (for Faster R-CNN)

### 7.2 Inference Phase Issues

**No Dog Detected**:
- Action: 
  - Lower confidence threshold and retry
  - Inform user: "No dog face detected. Please ensure the image contains a clear dog face."
  - Suggest retaking photo with better lighting/angle

**Multiple Dogs Detected**:
- Action:
  - Process all detected faces independently
  - Display separate bounding boxes and labels for each dog
  - Allow user to select which result to focus on

**Low Confidence Predictions** (< 0.6):
- Action:
  - Display confidence score alongside label
  - Show top-3 predictions with probabilities
  - Warn user: "Low confidence prediction. Result may be inaccurate."

**Poor Image Quality** (blurry, dark, overexposed):
- Action:
  - Implement image quality check before inference
  - Calculate sharpness score (Laplacian variance)
  - Calculate brightness histogram
  - If quality is poor, warn user and suggest retaking

**Edge Cases**:
- Puppy vs adult dog: Model may struggle with very young dogs
- Unusual breeds: May not generalize well to rare breeds
- Extreme angles: Profile views harder than frontal views
- Occlusions: Toys, hands, or other objects blocking face

### 7.3 Memory & Performance Optimization

**GPU Memory Management**:
- Clear CUDA cache after each inference: `torch.cuda.empty_cache()`
- Use `torch.no_grad()` during inference to disable gradient computation
- Process images sequentially to avoid memory buildup
- Monitor GPU memory usage: `torch.cuda.memory_allocated()`

**CPU Fallback**:
- If GPU unavailable, automatically fall back to CPU
- Warn user about slower performance (10-20x slower)
- Reduce batch size to 1 for CPU inference

---

## 8. Implementation Checklist
### 8.1 Development Phases
**Phase 1: Environment Setup** (Week 1)
- [ ] Set up AWS SageMaker JupyterLab with T4 GPU
- [ ] Install PyTorch, torchvision, ultralytics (for YOLOv8)
- [ ] Test GPU availability and memory
- [ ] Download and organize both datasets

**Phase 2: Detection Model Training** (Week 2-3)
- [ ] Preprocess detection dataset
- [ ] Implement data augmentation pipeline
- [ ] Train YOLOv8/Faster R-CNN with transfer learning
- [ ] Evaluate on validation set (target mAP > 0.7)
- [ ] Tune hyperparameters if needed
- [ ] Save best model

**Phase 3: Classification Model Training** (Week 4-5)
- [ ] Preprocess emotion dataset
- [ ] Implement data augmentation pipeline
- [ ] Train ResNet50 with two-phase training (frozen → fine-tune)
- [ ] Evaluate on validation set (target accuracy > 80%)
- [ ] Analyze confusion matrix for weak classes
- [ ] Save best model

**Phase 4: Pipeline Integration** (Week 6)
- [ ] Build unified inference pipeline
- [ ] Test on sample images
- [ ] Optimize inference speed
- [ ] Handle edge cases and errors
- [ ] Document results and metrics

### 8.2 Success Criteria
- Detection mAP@0.5 > 0.70
- Classification accuracy > 80%
- End-to-end inference time < 200ms per image (on T4 GPU)
- Robust handling of edge cases (no crashes)