# Detection Model Training Guide

This document provides comprehensive information about detection model architecture, experiment configurations, training strategies, and evaluation metrics for the Visual Dog Emotion Recognition system.

---

## 📊 Experiment Overview

The project now includes **3 diverse detection architectures** for comprehensive comparison:

| Experiment | Script | Architecture | Backbone | Key Feature |
|------------|--------|--------------|----------|-------------|
| Exp01: YOLOv8 Baseline | `exp01_detection_YOLOv8_baseline.py` | Single-stage (Anchor-based) | Medium (m) | Fast inference, balanced accuracy |
| Exp02: Faster R-CNN | `exp02_detection_Faster-RCNN.py` | Two-stage (Region Proposal) | ResNet50+FPN | Higher accuracy, slower inference |
| Exp03: SSD | `exp03_detection_SSD.py` | Single-stage (Multi-scale) | VGG16 | Moderate speed, good small object detection |

**Dataset**: Dog Face Detection Dataset  
**Preprocessing**: Images resized to 640×640, labels in YOLO/COCO/VOC format  
**Splits**: Train/Val/Test with stratification  
**Training**: Mixed precision (AMP), early stopping, ≥100 epochs

---

## 🏗️ Detection Model Architectures

### 1. YOLOv8Detector

**File**: [`src/models/detection_model.py`](src/models/detection_model.py)

```python
class YOLOv8Detector:
    """
    Single YOLOv8 wrapper with configurable parameters.
    
    Configuration options:
    - backbone_depth: 'n', 's', 'm', 'l', 'x' (model size)
    - input_size: 640, 1280, etc.
    - confidence_threshold: 0.3, 0.5, 0.7
    - nms_iou_threshold: 0.45, 0.5, 0.6
    - anchor_settings: custom anchor boxes
    """
```

**Key Features:**
- Unified wrapper around Ultralytics YOLOv8
- Configurable backbone depth for speed/accuracy trade-off
- Customizable inference thresholds
- Supports multi-dog detection

### Available Backbone Depths

| Depth | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| **nano (n)** | ~3.2M | Fastest | Lower | Edge devices, real-time |
| **small (s)** | ~11.2M | Fast | Good | Mobile deployment |
| **medium (m)** | ~25.9M | Moderate | Better | Balanced performance |
| **large (l)** | ~43.7M | Slow | High | Server deployment |
| **xlarge (x)** | ~68.2M | Slowest | Highest | Maximum accuracy |

### 2. FasterRCNNDetector

**File**: [`src/models/torchvision_detection.py`](src/models/torchvision_detection.py)

```python
class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN detector with ResNet50+FPN backbone.
    
    Two-stage detection:
    1. Region Proposal Network (RPN) generates candidate boxes
    2. ROI heads classify and refine boxes
    
    Characteristics:
    - Higher accuracy than single-stage detectors
    - Slower inference speed
    - Better for small objects
    """
```

**Key Features:**
- Two-stage detection paradigm (region proposal + classification)
- ResNet50 backbone with Feature Pyramid Network (FPN)
- Pretrained on COCO dataset
- Manually implemented training loop

**Training Configuration:**
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.005 (higher for SGD)
- Batch size: 4 (memory constraints on T4 GPU)
- Gradient accumulation: 4 steps (effective batch = 16)
- Epochs: 150 (longer for two-stage convergence)

### 3. SSDDetector

**File**: [`src/models/torchvision_detection.py`](src/models/torchvision_detection.py)

```python
class SSDDetector(nn.Module):
    """
    SSD (Single Shot Detector) with VGG16 backbone.
    
    Single-stage detection:
    - Predicts bounding boxes and class probabilities in one pass
    - Uses multi-scale feature maps for different object sizes
    
    Characteristics:
    - Moderate inference speed
    - Good balance between speed and accuracy
    - Effective for small objects via multi-scale predictions
    """
```

**Key Features:**
- Single-stage detection (direct prediction)
- VGG16 backbone with multi-scale feature maps
- Pretrained on COCO dataset
- Efficient memory usage

**Training Configuration:**
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.001
- Batch size: 16 (lighter architecture allows larger batches)
- Gradient accumulation: Not needed
- Epochs: 150

---

## ⚙️ Experiment Configurations

### Exp01: YOLOv8 Baseline

**Script**: [`experiments/exp01_detection_YOLOv8_baseline.py`](experiments/exp01_detection_YOLOv8_baseline.py)

**Model Configuration:**
```python
model_config = {
    'backbone': 'm',              # Medium backbone
    'input_size': 640,            # Standard resolution
    'confidence_threshold': 0.5,  # Moderate confidence
    'nms_iou_threshold': 0.45,    # Standard NMS
}
```

**Training Configuration:**
```python
training_config = {
    'epochs': 120,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 24,
    'use_amp': True,
    'early_stopping_patience': 0,  # Disabled
    'warmup_epochs': 10,
    'scheduler': 'cosine',
}
```

**Purpose**: Establish baseline performance for dog face detection  
**Expected**: Balanced speed and accuracy

### Exp02: Faster R-CNN

**Script**: [`experiments/exp02_detection_Faster-RCNN.py`](experiments/exp02_detection_Faster-RCNN.py)

**Model Configuration:**
```python
model_config = {
    'architecture': 'faster_rcnn',
    'backbone': 'resnet50_fpn',
    'num_classes': 2,  # background + dog_face
    'pretrained': True
}
```

**Training Configuration:**
```python
training_config = {
    'learning_rate': 0.005,           # SGD needs higher LR
    'batch_size': 4,                  # Small due to memory
    'epochs': 150,                    # Longer training
    'optimizer': 'sgd',               # SGD with momentum
    'weight_decay': 5e-4,             # Stronger regularization
    'early_stopping_patience': 20,
    'use_amp': True,
    'gradient_accumulation_steps': 4, # Effective batch = 16
}
```

**Purpose**: Test two-stage detection paradigm  
**Expected**: Higher accuracy but slower inference

### Exp03: SSD

**Script**: [`experiments/exp03_detection_SSD.py`](experiments/exp03_detection_SSD.py)

**Model Configuration:**
```python
model_config = {
    'architecture': 'ssd',
    'backbone': 'vgg16',
    'num_classes': 2,
    'pretrained': True
}
```

**Training Configuration:**
```python
training_config = {
    'learning_rate': 0.001,
    'batch_size': 16,                 # Larger batch possible
    'epochs': 150,
    'optimizer': 'sgd',
    'weight_decay': 5e-4,
    'early_stopping_patience': 20,
    'use_amp': True,
    'gradient_accumulation_steps': 1,
}
```

**Purpose**: Test single-stage multi-scale detection  
**Expected**: Moderate speed, good small object detection

---

## 🔄 Data Format Conversion

### Why Multiple Formats?

Different detection frameworks require different annotation formats:

| Framework | Format | File Type | Structure |
|-----------|--------|-----------|-----------|
| **YOLOv8** (Ultralytics) | YOLO | `.txt` | `class x_center y_center width height` (normalized) |
| **Faster R-CNN** (Torchvision) | COCO | `.json` | JSON with images, annotations, categories |
| **SSD** (Torchvision) | VOC | `.xml` | XML with bounding box coordinates |

### Conversion Script

**File**: [`src/data_processing/convert_detection_format.py`](src/data_processing/convert_detection_format.py)

```bash
# Convert to both COCO and VOC formats
python src/data_processing/convert_detection_format.py --format both

# Convert only to COCO (for Faster R-CNN)
python src/data_processing/convert_detection_format.py --format coco

# Convert only to VOC (for SSD)
python src/data_processing/convert_detection_format.py --format voc

# Custom source directory
python src/data_processing/convert_detection_format.py --source-dir data/processed/detection_small
```

**Output Structure:**
```
data/processed/
├── detection/                    # Original YOLO format
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset.yaml
│
├── detection_coco/               # COCO format (Faster R-CNN)
│   ├── images/{train,val,test}/  # Copied from original
│   ├── annotations/
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   └── dataset.yaml
│
└── detection_voc/                # VOC format (SSD)
    ├── images/{train,val,test}/  # Copied from original
    ├── annotations/{train,val,test}/  # XML files
    └── dataset.yaml
```

**Key Features:**
- Preserves original train/val/test split
- Only converts annotation format, not images
- Generates compatible `dataset.yaml` for each format
- Handles edge cases (empty labels, missing files)

---

## 📊 Model Comparison Matrix

### Architecture Comparison

| Feature | YOLOv8 | Faster R-CNN | SSD |
|---------|--------|--------------|-----|
| **Detection Paradigm** | Single-stage | Two-stage | Single-stage |
| **Backbone** | CSPDarknet | ResNet50+FPN | VGG16 |
| **Parameters** | ~25.9M | ~41M | ~26M |
| **Feature Pyramid** | Built-in (PANet) | FPN | Multi-scale conv |
| **Anchor Strategy** | Anchor-free | Anchor-based | Anchor-based |
| **Pretrained** | COCO | COCO | COCO |

### Performance Expectations (T4 GPU)

| Metric | YOLOv8 | Faster R-CNN | SSD |
|--------|--------|--------------|-----|
| **Training Speed** | ⭐⭐⭐ Fastest (~2-3h) | ⭐ Slowest (~4-5h) | ⭐⭐ Moderate (~3-4h) |
| **Inference Speed** | ⭐⭐⭐ ~40-60 FPS | ⭐ ~10-15 FPS | ⭐⭐ ~25-35 FPS |
| **mAP@0.5** | ⭐⭐ ~0.75-0.80 | ⭐⭐⭐ ~0.80-0.85 | ⭐⭐ ~0.72-0.78 |
| **Small Objects** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐ Good |
| **Memory Usage** | ⭐⭐ Moderate | ⭐ Highest | ⭐⭐ Low |
| **Batch Size (T4)** | 24 | 4 (+ grad accum) | 16 |

### Training Configuration Comparison

| Config | YOLOv8 | Faster R-CNN | SSD |
|--------|--------|--------------|-----|
| **Optimizer** | Adam | SGD + Momentum | SGD + Momentum |
| **Learning Rate** | 0.001 | 0.005 | 0.001 |
| **Weight Decay** | 1e-4 | 5e-4 | 5e-4 |
| **Epochs** | 120 | 150 | 150 |
| **Effective Batch** | 24 | 16 (4×4) | 16 |
| **AMP** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Early Stopping** | Disabled | Patience=20 | Patience=20 |
| **LR Scheduler** | Cosine | Cosine | Cosine |

### When to Use Each Model

| Scenario | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Real-time application** | YOLOv8 | Fastest inference, good accuracy |
| **Maximum accuracy needed** | Faster R-CNN | Two-stage refinement, best mAP |
| **Limited GPU memory** | SSD or YOLOv8 | Lower memory footprint |
| **Small object detection** | Faster R-CNN | FPN + two-stage excels at small objects |
| **Mobile deployment** | YOLOv8 (nano/small) | Optimized for edge devices |
| **Research/academic study** | All three | Comprehensive paradigm comparison |
| **Production server** | YOLOv8 or SSD | Balance of speed and accuracy |
| **Educational purposes** | Start with YOLOv8, then others | Learn evolution of detection |

---

## 📈 Detection Training Strategies

### Built-in YOLOv8 Augmentations

YOLOv8 includes powerful built-in data augmentations:

1. **Mosaic Augmentation**: Combines 4 images into one, improves context understanding
2. **Mixup**: Blends two images and their labels
3. **Random Horizontal Flip**: Mirrors images horizontally
4. **HSV Color Space Augmentation**: Randomly adjusts hue, saturation, value
5. **Perspective Transform**: Applies random perspective warping
6. **Scale Augmentation**: Randomly scales images
7. **Translation**: Randomly shifts images

**Note**: These augmentations are automatically applied during training and don't require manual configuration in our code.

### Learning Rate Strategy

**Default Configuration:**
- Initial LR: 0.001
- Optimizer: Adam (adaptive learning rate)
- No explicit scheduler (YOLOv8 uses internal LR scheduling)

**Customization Options:**
```python
# For faster convergence
training_config = {
    'learning_rate': 0.01,        # Higher initial LR
    'optimizer': 'SGD',           # Traditional optimizer
    'momentum': 0.937,            # YOLOv8 default momentum
}

# For more stable training
training_config = {
    'learning_rate': 0.0001,      # Lower initial LR
    'optimizer': 'AdamW',         # Better weight decay handling
}
```

### Regularization Techniques

1. **Weight Decay**: Applied through optimizer (default 5e-4)
2. **Dropout**: Built into YOLOv8 architecture
3. **Data Augmentation**: Extensive built-in augmentations
4. **Early Stopping**: Patience 15 epochs to prevent overfitting
5. **Mixed Precision**: AMP for numerical stability

---

## 📊 Detection Evaluation Metrics

### Comprehensive Metrics Suite

The evaluation framework now provides professional-grade detection metrics:

#### 1. Mean Average Precision (mAP)

**mAP@0.5**: Average Precision at IoU threshold = 0.5
- Primary metric for most detection tasks
- Measures detection accuracy with moderate overlap requirement

**mAP@0.5:0.95**: Average mAP across IoU thresholds from 0.5 to 0.95 (step 0.05)
- More stringent metric (COCO standard)
- Evaluates localization precision across multiple thresholds
- Formula: `mean(AP@0.5, AP@0.55, AP@0.6, ..., AP@0.95)`

#### 2. Per-Class Metrics

For each class, calculates:
- **Average Precision (AP@0.5)**: Area under Precision-Recall curve
- **Precision**: TP / (TP + FP) - How many detected dogs are correct?
- **Recall**: TP / (TP + FN) - How many actual dogs were detected?
- **F1-Score**: Harmonic mean of precision and recall

#### 3. Overall Metrics

- **Precision**: Overall precision across all classes at IoU=0.5
- **Recall**: Overall recall across all classes at IoU=0.5
- **F1-Score**: Balanced measure of precision and recall

#### 4. IoU Distribution Analysis

Statistical analysis of Intersection-over-Union values:
- **Mean IoU**: Average localization quality
- **Median IoU**: Robust central tendency
- **Standard Deviation**: Consistency of detections
- **Min/Max IoU**: Range of localization quality
- **Histogram**: Distribution visualization (20 bins)

#### 5. mAP vs IoU Threshold Curve

Shows how mAP changes with different IoU thresholds:
- Helps understand model's localization precision
- Steeper decline indicates less precise bounding boxes
- Flatter curve indicates consistent localization quality

### Visualization Outputs

The evaluator generates comprehensive visualizations:

1. **IoU_distribution.png**: Histogram of IoU values for matched detections
   - Shows distribution of localization quality
   - Mean and median marked with vertical lines

2. **mAP_vs_IoU.png**: mAP at different IoU thresholds
   - Line plot showing AP degradation as IoU increases
   - Helps identify localization precision issues

3. **per_class_metrics.png**: Bar chart of per-class metrics
   - Side-by-side comparison of Precision, Recall, F1
   - Identifies class-specific performance differences

### Implementation Details

**Average Precision Calculation**:
- Uses 11-point interpolation method (PASCAL VOC standard)
- Matches predictions to ground truths using greedy algorithm
- Sorts predictions by confidence score (descending)
- Calculates cumulative precision and recall

**Matching Strategy**:
- For each prediction, finds best matching ground truth (highest IoU)
- Each ground truth can only be matched once
- Predictions with IoU < threshold are false positives
- Unmatched ground truths are false negatives

**Key Files**:
- [`src/evaluation/detection_evaluator.py`](src/evaluation/detection_evaluator.py)

---

## 📂 Detection Output Organization

Each detection experiment saves outputs to timestamped directories:

```
outputs/exp01_detection_YOLOv8_baseline/
└── run_20260420_193045/
    ├── model/
    │   ├── best_model.pt        ← Best model weights
    │   └── model_config.json   ← Model configuration used
    │
    ├── logs/
    │   ├── training_log.csv    ← Epoch-by-epoch metrics
    │   ├── evaluation_metrics.json ← Comprehensive test metrics
    │   └── experiment_report.md ← Full markdown report
    │
    └── figures/
        ├── IoU_distribution.png      ← IoU histogram with statistics
        ├── mAP_vs_IoU.png            ← mAP at different IoU thresholds
        └── per_class_metrics.png     ← Per-class precision/recall/F1
```

### Output Contents

#### Model Folder
- `best_model.pt`: Trained model weights (best validation loss/mAP)
- `final_model.pt`: Final model weights after training completion
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```csv
  epoch, train_loss, val_loss, learning_rate
  1, 2.345, 2.123, 0.001
  2, 1.987, 1.876, 0.001
  ...
  ```
- `evaluation_metrics.json`: Comprehensive test set metrics including:
  ```json
  {
    "mAP50": 0.7856,
    "mAP50_95": 0.6234,
    "precision": 0.8123,
    "recall": 0.7654,
    "f1_score": 0.7882,
    "per_class_metrics": {
      "0": {
        "AP50": 0.7856,
        "precision": 0.8123,
        "recall": 0.7654,
        "f1_score": 0.7882
      }
    },
    "ap_at_iou": {
      "0.5": 0.7856,
      "0.55": 0.7234,
      ...
      "0.95": 0.3456
    },
    "iou_statistics": {
      "mean": 0.6789,
      "median": 0.7012,
      "std": 0.1234,
      "min": 0.5001,
      "max": 0.9876
    }
  }
  ```
- `experiment_report.md`: Comprehensive markdown report including:
  - Overall metrics table
  - Per-class metrics table
  - IoU statistics table
  - Figure references
  - Interpretation guidelines

#### Figures Folder
- `IoU_distribution.png`: Histogram of IoU values with mean/median markers
- `mAP_vs_IoU.png`: Line plot showing mAP degradation across IoU thresholds
- `per_class_metrics.png`: Bar chart comparing precision, recall, F1 per class

---

## 🚀 Running Detection Experiments

### Single Experiment

```bash
# Detection baseline
python experiments/exp01_detection_YOLOv8_baseline.py

# With custom parameters
python experiments/exp01_detection_YOLOv8_baseline.py --lr 0.001 --batch_size 16 --epochs 120
```

### Quick Testing with Small Subset

```bash
# Run with subset flag for quick validation
python experiments/exp01_detection_YOLOv8_baseline.py --use-small-subset
```

### Backbone Comparison

To compare different backbones:

```python
# In exp01_detection_YOLOv8_baseline.py
model_config = {
    'backbone': 's',  # Try 'n', 's', 'm', 'l', 'x'
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
}
```

---

## 💡 Detection Best Practices

### 1. Reproducibility

- Fixed random seeds in all experiments:
  ```python
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  ```
- Save complete configuration to `model_config.json`
- Log all hyperparameters and environmental info

### 2. Resource Efficiency

- Use mixed precision training (`use_amp=True`)
- Adjust batch size based on available GPU memory
- Choose appropriate backbone for your hardware:
  - CPU/Low-end GPU: Use 'n' or 's'
  - Mid-range GPU: Use 'm' ← **Recommended**
  - High-end GPU: Use 'l' or 'x'
- Monitor GPU memory usage: `nvidia-smi`

### 3. Debugging Tips

**Problem**: Out of Memory (OOM)
```
Solution:
- Reduce batch_size (e.g., 16 → 8)
- Use smaller backbone ('m' → 's')
- Reduce input_size (640 → 320)
- Enable AMP if not already enabled
```

**Problem**: Low mAP (< 0.5)
```
Solution:
- Check label quality (verify YOLO format)
- Increase training epochs
- Try larger backbone ('m' → 'l')
- Increase input_size (640 → 1280)
- Verify data augmentation is working
```

**Problem**: Too many false positives
```
Solution:
- Increase confidence_threshold (0.5 → 0.7)
- Decrease nms_iou_threshold (0.45 → 0.3)
- Train longer with more epochs
- Add more diverse training data
```

**Problem**: Missed detections (low recall)
```
Solution:
- Decrease confidence_threshold (0.5 → 0.3)
- Increase input_size for better small object detection
- Use larger backbone for better feature extraction
- Check if training data has sufficient examples
```

### 4. Monitoring Training

**Real-time monitoring:**
```bash
# Watch training log
tail -f outputs/<experiment>/run_<timestamp>/logs/training_log.csv

# Monitor GPU usage
watch -n 1 nvidia-smi
```

**Visualization:**
- Open generated figures in `figures/` directory
- Read comprehensive report in `experiment_report.md`

---

## 🔧 Advanced Configuration

### Custom Anchor Boxes

If detecting dogs at unusual scales, you can customize anchor boxes:

```python
model_config = {
    'backbone': 'm',
    'input_size': 640,
    'anchor_settings': {
        'small': [10, 13, 16, 30, 33, 23],
        'medium': [30, 61, 62, 45, 59, 119],
        'large': [116, 90, 156, 198, 373, 326]
    }
}
```

### Multi-Scale Training

Train on multiple input sizes for robustness:

```python
training_config = {
    'epochs': 120,
    'multi_scale': True,          # Enable multi-scale training
    'scale_range': [320, 640],    # Range of input sizes
}
```

### Transfer Learning

Fine-tune on domain-specific data:

```python
# Load pretrained COCO weights
model = YOLOv8Detector(backbone='m', pretrained=True)

# Freeze backbone initially
for param in model.backbone.parameters():
    param.requires_grad = False

# Train head only for 10 epochs
# Then unfreeze and fine-tune with lower LR
```

---

## 📈 Performance Expectations

Based on backbone selection:

| Backbone | Expected mAP@0.5 | Inference Speed (FPS) | Model Size | Best Use Case |
|----------|------------------|----------------------|------------|---------------|
| **nano (n)** | 0.60-0.70 | 100+ FPS | ~6 MB | Real-time mobile apps |
| **small (s)** | 0.70-0.80 | 50-80 FPS | ~22 MB | Mobile deployment |
| **medium (m)** | 0.80-0.85 | 20-40 FPS | ~52 MB | Balanced performance ← **Recommended** |
| **large (l)** | 0.85-0.90 | 10-20 FPS | ~88 MB | Server deployment |
| **xlarge (x)** | 0.90-0.95 | 5-10 FPS | ~140 MB | Maximum accuracy |

**Note**: Actual performance depends on dataset quality, training duration, and hardware capabilities.

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Label Format Errors**

**Symptom**: Training fails with label parsing errors

**Solution**:
- Verify labels are in YOLO format: `class x_center y_center width height`
- All values should be normalized to [0, 1]
- Check for empty label files: `find . -name "*.txt" -empty`
- Validate coordinates don't exceed image boundaries

**Issue 2: Poor Convergence**

**Symptom**: Loss doesn't decrease after 20+ epochs

**Solution**:
- Check learning rate (try 0.001 → 0.01 or 0.0001)
- Verify data loading is correct
- Increase batch size if possible
- Check for label noise or incorrect annotations

**Issue 3: Overfitting**

**Symptom**: Train mAP >> Val mAP (gap > 0.15)

**Solution**:
- Enable stronger augmentations
- Increase early stopping patience
- Add dropout or weight decay
- Collect more diverse training data

---

**Last Updated**: 2026-04-26  
**Implemented by**: AI Assistant  
**Status**: ✅ Documentation created
