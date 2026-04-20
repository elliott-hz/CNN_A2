# 42028: Deep Learning and Convolutional Neural Network
## Dataset details, GUI design, Implementation Plan
**Project Title and number**: Visual Dog Emotion Recognition

---

## 1. Dataset Details
This project uses two public datasets to support the two main computer vision tasks: **object detection** and **image classification**.

### 1.1 Object Detection Task
- Dataset: Dog Face Detection dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/wutheringwang/dog-face-detectionyolo-format/data
- Images: ~6,000
- Source: Stanford Dogs Dataset + Oxford-IIIT Pet Dataset
- Purpose: Detect and localize dog faces in images/video frames

### 1.2 Image Classification Task
- Dataset: Dog Emotions – 5 Classes dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/dougandrade/dog-emotions-5-classes/data
- Images: ~9,000 (≈1,800 per class)
- 5 Emotion Classes: angry, happy, relaxed, frown, alert
- Purpose: Classify emotion of detected dog faces

### Train/Validation/Test Split
70% / 20% / 10% for both datasets.

---

## 2. GUI Design
### Figure 1. Start-up interface
- Main interface with: **Dog detection**, **Emotion detection**, **Browse** (upload image/video)
- Emotion detection only available after valid dog input is uploaded
- Ensures logical workflow and data relevance

### Figure 2. Object detection result (after upload)
- System detects dog with bounding box + label
- Emotion Detection button becomes enabled
- Visual feedback confirms successful detection

### Figure 3. Emotion detection output
- After detection, CNN classifies emotion (e.g., happy)
- Result shown directly on the image
- Complete end-to-end pipeline demonstrated

---

## 3. Implementation Plan
### 3.1 CNN Architectures to Explore
#### Detection Task
- YOLO, SSD, Faster R-CNN
- Compare accuracy, stability, speed

#### Classification Task
- AlexNet, GoogLeNet, VGG, ResNet, MobileNet
- Evaluate multi-class emotion recognition performance
- Select one detection + one classification model for final system

### 3.2 Training Strategies
- **Training**: Train from scratch
- **Image Preprocessing**:
  - Detection: random crop, flip, scale, color adjustment
  - Classification: resize, normalize, rotation, brightness, noise injection
- **Optimizer**: Adam, SGD with momentum, RMSprop
- **Learning rate**: step decay, cosine annealing
- **Early stopping**: prevent overfitting
- **Hyperparameter tuning**: grid search (learning rate, batch size, layers, dropout)
- **Loss function**:
  - Classification: cross-entropy, weighted cross-entropy, focal loss
  - Detection: standard detection loss + optional focal loss
- **Regularization**: dropout, batch normalization, weight decay

### 3.3 Testing Plan
#### Model-level evaluation
- Detection: mAP, IoU (no dog, single, multiple, partial faces)
- Classification: accuracy, precision, recall, F1-score, confusion matrix

#### System-level evaluation
- Test on images, videos, livestream
- Measure detection/classification accuracy, speed, stability

### 3.4 Error Handling Plan
- Overfitting: augmentation, dropout, early stopping
- Underfitting: adjust model size + training time
- Detection errors: better preprocessing, tuning, confidence threshold
- Classification errors: resampling, class weighting, confusion matrix analysis
- Real-time performance: optimize resolution + model size, temporal smoothing

### 3.5 Model Integration Strategy
Two-stage pipeline:
1. Detect dog faces
2. Classify emotion for each detected region

### 3.6 Tentative Timeline
- Week 5–7: Dataset prep, model selection, implementation plan (Part B)
- Week 7–9: Train models, initial experiments (Part C)
- Week 9–11: System integration + GUI development (Part D)
- Week 11–13: Optimization, testing, final submission (Part E)
