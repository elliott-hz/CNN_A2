# Visual Dog Emotion Recognition - Project Structure Design
*Fully aligned with Assignment 3 PartC | Simplified Architecture | 2 Datasets • 2 Models • 6 Experiments*

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Module Descriptions](#3-module-descriptions)
4. [Data Flow](#4-data-flow)
5. [Experiment Workflow](#5-experiment-workflow)
6. [Output Organization](#6-output-organization)
7. [Key Design Principles](#7-key-design-principles)

---

## 1. Project Overview

This project implements a **two-stage deep learning pipeline** for visual dog emotion recognition:
- **Stage 1**: Dog face detection using YOLOv8
- **Stage 2**: Emotion classification using ResNet50

### Core Architecture
```
Input Image/Video
       ↓
┌─────────────────────┐
│  Dog Face Detection  │ ← YOLOv8 (configurable variants)
│  (Bounding Box)      │
└─────────────────────┘
       ↓ (crop face)
┌─────────────────────┐
│ Emotion Classification│ ← ResNet50 (configurable variants)
│  (5 emotions)        │
└─────────────────────┘
       ↓
Output: BBox + Emotion Label
```

### Technical Stack
- **Framework**: PyTorch 2.0+ with torchvision
- **Hardware**: AWS SageMaker JupyterLab with NVIDIA T4 GPU (16GB VRAM)
- **Detection Model**: YOLOv8 (single base model with configurable parameters)
- **Classification Model**: ResNet50 (single base model with configurable parameters)
- **Datasets**: 
  - Dog Face Detection Dataset (~6,000 images)
  - Dog Emotion Dataset (~9,000 images, 5 classes)

---

## 2. Directory Structure

```
CNN_A3/
│
├── README.md                          # Project overview and usage guide
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Global configuration (paths, defaults)
├── PartC-Project Structure.md         # This file
│
├── data/                              # Data directory (auto-created)
│   ├── raw/                           # Original downloaded datasets
│   │   ├── detection_dataset/         # Dog Face Detection (Kaggle)
│   │   └── emotion_dataset/           # Dog Emotion (Kaggle)
│   │
│   └── processed/                     # Preprocessed & split datasets
│       ├── detection/                 # Detection splits
│       │   ├── X_train.npy
│       │   ├── X_valid.npy
│       │   ├── X_test.npy
│       │   ├── y_train.npy
│       │   ├── y_valid.npy
│       │   ├── y_test.npy
│       │   └── metadata.json          # Split statistics
│       │
│       └── emotion/                   # Emotion splits
│           ├── X_train.npy
│           ├── X_valid.npy
│           ├── X_test.npy
│           ├── y_train.npy
│           ├── y_valid.npy
│           ├── y_test.npy
│           └── metadata.json
│
├── src/                               # Source code package
│   ├── __init__.py
│   │
│   ├── data_processing/               # Data download & preprocessing
│   │   ├── __init__.py
│   │   ├── download_datasets.py       # Download datasets (run once)
│   │   ├── detection_preprocessor.py  # Detection data preprocessing
│   │   ├── emotion_preprocessor.py    # Emotion data preprocessing
│   │   ├── augmentation.py            # Data augmentation utilities
│   │   └── dataset_utils.py           # Common utilities
│   │
│   ├── models/                        # Model definitions (SIMPLIFIED)
│   │   ├── __init__.py
│   │   ├── detection_model.py         # ONE model: YOLOv8Detector
│   │   │   └── class YOLOv8Detector   # Configurable via parameters
│   │   │
│   │   └── classification_model.py    # ONE model: ResNet50Classifier
│   │       └── class ResNet50Classifier # Configurable via parameters
│   │
│   ├── training/                      # Training frameworks
│   │   ├── __init__.py
│   │   ├── detection_trainer.py       # Detection training logic
│   │   │   ├── class DetectionTrainer
│   │   │   ├── init_model()
│   │   │   ├── train_epoch()
│   │   │   ├── validate()
│   │   │   └── optimize()
│   │   │
│   │   └── classification_trainer.py  # Classification training logic
│   │       ├── class ClassificationTrainer
│   │       ├── init_model()
│   │       ├── train_epoch()
│   │       ├── validate()
│   │       └── optimize()
│   │
│   ├── evaluation/                    # Evaluation frameworks
│   │   ├── __init__.py
│   │   ├── detection_evaluator.py     # Detection metrics (mAP, IoU)
│   │   │   ├── calculate_mAP()
│   │   │   ├── calculate_IoU()
│   │   │   ├── plot_precision_recall()
│   │   │   └── generate_evaluation_report()
│   │   │
│   │   └── classification_evaluator.py # Classification metrics
│   │       ├── calculate_metrics()
│   │       ├── plot_confusion_matrix()
│   │       ├── plot_roc_curve()
│   │       └── generate_evaluation_report()
│   │
│   ├── inference/                     # Inference pipeline
│   │   ├── __init__.py
│   │   ├── detection_inference.py     # Detection-only inference
│   │   ├── classification_inference.py # Classification-only inference
│   │   └── pipeline_inference.py      # End-to-end stacked inference
│   │       ├── detect_and_classify()
│   │       └── visualize_results()
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       ├── visualization.py           # Plotting utilities
│       ├── metrics.py                 # Metric calculations
│       └── file_utils.py              # File I/O helpers
│
├── experiments/                       # 6 Experiment scripts
│   ├── __init__.py
│   ├── exp01_detection_baseline.py        # YOLOv8 baseline
│   ├── exp02_detection_modified_v1.py     # YOLOv8 modified v1
│   ├── exp03_detection_modified_v2.py     # YOLOv8 modified v2
│   ├── exp04_classification_baseline.py   # ResNet50 baseline
│   ├── exp05_classification_modified_v1.py # ResNet50 modified v1
│   ├── exp06_classification_modified_v2.py # ResNet50 modified v2
│   │
│   └── experiment_template.py         # Template for new experiments
│
├── outputs/                           # Experiment outputs (SIMPLIFIED)
│   ├── exp01_detection_baseline/      # Experiment 1 folder
│   │   ├── run_20260420_193045/       # Run 1 (timestamped)
│   │   │   ├── model/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── model_config.json
│   │   │   ├── logs/
│   │   │   │   ├── training_log.csv
│   │   │   │   └── experiment_report.md
│   │   │   └── figures/
│   │   │       ├── precision_recall_curve.png
│   │   │       ├── IoU_distribution.png
│   │   │       └── sample_detections.png
│   │   │
│   │   └── run_20260421_101523/       # Run 2 (new timestamp)
│   │       └── ... (same structure)
│   │
│   ├── exp02_detection_modified_v1/   # Experiment 2 folder
│   │   └── run_TIMESTAMP/
│   │       ├── model/
│   │       ├── logs/
│   │       └── figures/
│   │
│   ├── exp03_detection_modified_v2/   # Experiment 3 folder
│   │   └── run_TIMESTAMP/
│   │       └── ...
│   │
│   ├── exp04_classification_baseline/ # Experiment 4 folder
│   │   └── run_TIMESTAMP/
│   │       └── ...
│   │
│   ├── exp05_classification_modified_v1/ # Experiment 5 folder
│   │   └── run_TIMESTAMP/
│   │       └── ...
│   │
│   └── exp06_classification_modified_v2/ # Experiment 6 folder
│       └── run_TIMESTAMP/
│           └── ...
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_inference_demo.ipynb
│
└── scripts/                           # Convenience scripts
    ├── download_data.sh               # Download datasets
    ├── run_all_experiments.sh         # Run all 6 experiments
    ├── run_single_experiment.sh       # Run specific experiment
    └── inference_demo.sh              # Demo inference
```

---

## 3. Module Descriptions

### 3.1 Data Processing (`src/data_processing/`)

#### Purpose
Handle dataset download, preprocessing, and splitting into train/valid/test sets.

#### Key Features
- **One-time download**: Checks if data exists before downloading
- **Unified output format**: Both datasets converted to numpy arrays
- **Persistent storage**: Processed data saved for reuse
- **Different preprocessing pipelines**:
  - Detection: Handles bounding box annotations, converts to YOLO format
  - Emotion: Handles folder-based classes, resizes to 224×224

#### Files
- `download_datasets.py`: Downloads both datasets from Kaggle
- `detection_preprocessor.py`: Preprocesses detection dataset
- `emotion_preprocessor.py`: Preprocesses emotion dataset
- `augmentation.py`: Data augmentation transforms
- `dataset_utils.py`: Common utilities (normalization, loading)

---

### 3.2 Model Definitions (`src/models/`)

#### **Simplified Design Philosophy**
Each task uses **ONE base model class** with **configurable parameters** to create different variants across experiments.

#### Detection Model: `detection_model.py`
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
    def __init__(self, config: dict):
        self.config = config
        # Initialize YOLOv8 with specified parameters
```

**Experiments using this model:**
- Exp01: Baseline (default params: backbone='m', input_size=640, conf=0.5)
- Exp02: Modified v1 (different params: backbone='l', input_size=1280, conf=0.6)
- Exp03: Modified v2 (different params: backbone='s', custom anchors, conf=0.4)

#### Classification Model: `classification_model.py`
```python
class ResNet50Classifier:
    """
    Single ResNet50 wrapper with configurable parameters.
    
    Configuration options:
    - dropout_rate: 0.3, 0.5, 0.7
    - additional_fc_layers: True/False
    - freeze_strategy: 'all', 'partial', 'none'
    - num_classes: 5 (fixed for this task)
    - use_batch_norm: True/False
    """
    def __init__(self, config: dict):
        self.config = config
        # Initialize ResNet50 with specified parameters
```

**Experiments using this model:**
- Exp04: Baseline (default params: dropout=0.5, freeze='partial')
- Exp05: Modified v1 (different params: dropout=0.7, additional layers)
- Exp06: Modified v2 (different params: dropout=0.3, no freeze)

---

### 3.3 Training Frameworks (`src/training/`)

#### Detection Trainer: `detection_trainer.py`
```python
class DetectionTrainer:
    """
    Training framework for detection models.
    
    Responsibilities:
    - Initialize model with config
    - Forward/backward computation
    - Optimizer management (SGD, Adam, AdamW)
    - Loss calculation (CIoU, BCE)
    - Validation with mAP monitoring
    - Early stopping
    - Mixed precision training (AMP)
    - Gradient accumulation
    """
    def __init__(self, model_config, training_config):
        pass
    
    def train(self, train_loader, val_loader):
        # Main training loop with progress bars
        pass
```

#### Classification Trainer: `classification_trainer.py`
```python
class ClassificationTrainer:
    """
    Training framework for classification models.
    
    Responsibilities:
    - Initialize model with config
    - Two-phase training (frozen → fine-tune)
    - Forward/backward computation
    - Optimizer management
    - Loss calculation (Cross-entropy with label smoothing)
    - Validation with accuracy monitoring
    - Early stopping
    - Class weighting for imbalance
    """
    def __init__(self, model_config, training_config):
        pass
    
    def train(self, train_loader, val_loader):
        # Main training loop with progress bars
        pass
```

#### Configurable Training Parameters
Both trainers accept these arguments:
- `learning_rate`: Initial learning rate
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `optimizer`: 'sgd', 'adam', 'adamw'
- `weight_decay`: L2 regularization strength
- `early_stopping_patience`: Patience for early stopping
- `use_amp`: Enable mixed precision training
- `gradient_accumulation_steps`: Steps for gradient accumulation
- `scheduler`: Learning rate scheduler type

---

### 3.4 Evaluation Frameworks (`src/evaluation/`)

#### Detection Evaluator: `detection_evaluator.py`
```python
class DetectionEvaluator:
    """
    Evaluation metrics for detection models.
    
    Metrics:
    - mAP@0.5 (primary)
    - mAP@0.5:0.95 (strict)
    - IoU distribution
    - Precision-Recall curves
    - Per-class AP
    """
    def evaluate(self, model, test_loader):
        # Calculate all metrics
        # Generate plots
        # Save report
        pass
```

#### Classification Evaluator: `classification_evaluator.py`
```python
class ClassificationEvaluator:
    """
    Evaluation metrics for classification models.
    
    Metrics:
    - Overall accuracy
    - Per-class Precision, Recall, F1-score
    - Confusion matrix
    - ROC curves (one-vs-rest)
    - Class-wise performance analysis
    """
    def evaluate(self, model, test_loader):
        # Calculate all metrics
        # Generate plots
        # Save report
        pass
```

---

### 3.5 Inference Pipeline (`src/inference/`)

Three inference modes:

#### 1. Detection-Only Inference
```python
from src.inference.detection_inference import DetectionInference

inference = DetectionInference(model_path='path/to/model.pt')
results = inference.predict(image_path)
# Output: List of bounding boxes with confidence scores
```

#### 2. Classification-Only Inference
```python
from src.inference.classification_inference import ClassificationInference

inference = ClassificationInference(model_path='path/to/model.pth')
results = inference.predict(cropped_face_image)
# Output: Emotion label + probability distribution
```

#### 3. Stacked Pipeline Inference (End-to-End)
```python
from src.inference.pipeline_inference import PipelineInference

pipeline = PipelineInference(
    detection_model='path/to/detection.pt',
    classification_model='path/to/classification.pth'
)
results = pipeline.predict(image_path)
# Output: Bounding boxes + emotion labels for each detected dog
```

---

### 3.6 Utilities (`src/utils/`)

- `logger.py`: Centralized logging setup
- `visualization.py`: Plotting functions (confusion matrix, ROC, PR curves)
- `metrics.py`: Metric calculation helpers
- `file_utils.py`: File I/O, directory creation, path management

---

## 4. Data Flow

### 4.1 Data Preparation Phase
```
┌─────────────────────────────────────┐
│  1. Download Datasets (Once)        │
│     - detection_dataset/            │
│     - emotion_dataset/              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Preprocess & Split              │
│     - Detection:                    │
│       * Parse annotations           │
│       * Augment (Mosaic, Flip)      │
│       * Split: 70/20/10             │
│       * Save as .npy files          │
│                                     │
│     - Emotion:                      │
│       * Load from folders           │
│       * Resize to 224×224           │
│       * Augment (Rotate, Flip)      │
│       * Split: 70/20/10             │
│       * Save as .npy files          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. Processed Data Ready            │
│     data/processed/detection/       │
│     data/processed/emotion/         │
│     (Reusable for all experiments)  │
└─────────────────────────────────────┘
```

### 4.2 Training Phase (Per Experiment)
```
┌─────────────────────────────────────┐
│  1. Load Preprocessed Data          │
│     (Skip if already processed)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Initialize Model                │
│     - Load base model class         │
│     - Apply experiment config       │
│     - Transfer learning weights     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. Configure Training              │
│     - Optimizer (SGD/Adam/AdamW)    │
│     - Loss function                 │
│     - Hyperparameters               │
│     - Scheduler                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  4. Train Model                     │
│     - Forward pass                  │
│     - Backward pass                 │
│     - Optimization step             │
│     - Validation every N epochs     │
│     - Early stopping check          │
│     - Progress bars (tqdm)          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  5. Evaluate on Test Set            │
│     - Calculate metrics             │
│     - Generate plots                │
│     - Create report                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  6. Save Outputs                    │
│     - Model weights                 │
│     - Training logs                 │
│     - Figures                       │
│     - Markdown report               │
└─────────────────────────────────────┘
```

---

## 5. Experiment Workflow

### 5.1 Six Experiments Overview

| Experiment | Task | Model | Variant | Key Differences |
|------------|------|-------|---------|-----------------|
| **Exp01** | Detection | YOLOv8 | Baseline | Default params (backbone='m', size=640) |
| **Exp02** | Detection | YOLOv8 | Modified v1 | Larger model (backbone='l', size=1280) |
| **Exp03** | Detection | YOLOv8 | Modified v2 | Smaller model + custom anchors |
| **Exp04** | Classification | ResNet50 | Baseline | Default params (dropout=0.5, partial freeze) |
| **Exp05** | Classification | ResNet50 | Modified v1 | Higher dropout + additional FC layers |
| **Exp06** | Classification | ResNet50 | Modified v2 | Lower dropout + no freeze |

### 5.2 Example Experiment Script Structure

```python
# experiments/exp01_detection_baseline.py

import sys
sys.path.append('..')

from src.data_processing.download_datasets import download_datasets
from src.data_processing.detection_preprocessor import DetectionPreprocessor
from src.models.detection_model import YOLOv8Detector
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.logger import setup_logger
from datetime import datetime
import os

def main():
    # 1. Setup logging
    logger = setup_logger('exp01_detection_baseline')
    
    # 2. Check/download data
    download_datasets()
    
    # 3. Preprocess (skip if exists)
    preprocessor = DetectionPreprocessor()
    if not preprocessor.is_processed():
        preprocessor.process()
    
    # 4. Load data
    train_data = preprocessor.load_split('train')
    val_data = preprocessor.load_split('valid')
    test_data = preprocessor.load_split('test')
    
    # 5. Define model config (BASELINE)
    model_config = {
        'backbone': 'm',
        'input_size': 640,
        'confidence_threshold': 0.5,
        'nms_iou_threshold': 0.45
    }
    
    # 6. Define training config
    training_config = {
        'learning_rate': 1e-3,
        'batch_size': 16,
        'epochs': 50,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'use_amp': True,
        'gradient_accumulation_steps': 1
    }
    
    # 7. Initialize model
    model = YOLOv8Detector(model_config)
    
    # 8. Initialize trainer
    trainer = DetectionTrainer(model_config, training_config)
    
    # 9. Create output directory (timestamped)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/exp01_detection_baseline/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 10. Train
    logger.info("Starting training...")
    trainer.train(train_data, val_data, output_dir)
    
    # 11. Evaluate
    logger.info("Starting evaluation...")
    evaluator = DetectionEvaluator()
    evaluator.evaluate(model, test_data, output_dir)
    
    # 12. Generate report
    evaluator.generate_report(output_dir)
    
    logger.info(f"Experiment completed. Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
```

### 5.3 Running Experiments

#### Option 1: Run Single Experiment
```bash
cd experiments
python exp01_detection_baseline.py
```

#### Option 2: Run All Experiments Sequentially
```bash
bash scripts/run_all_experiments.sh
```

#### Option 3: Run with Custom Parameters
```bash
python exp01_detection_baseline.py --lr 0.001 --batch_size 32 --epochs 100
```

---

## 6. Output Organization

### 6.1 Simplified Output Structure

Each experiment has its own folder. Each run creates a timestamped sub-folder containing all outputs.

```
outputs/
├── exp01_detection_baseline/
│   ├── run_20260420_193045/       ← First run
│   │   ├── model/
│   │   │   ├── best_model.pt      ← Best model weights
│   │   │   └── model_config.json  ← Model configuration used
│   │   │
│   │   ├── logs/
│   │   │   ├── training_log.csv   ← Epoch-by-epoch metrics
│   │   │   └── experiment_report.md ← Full markdown report
│   │   │
│   │   └── figures/
│   │       ├── precision_recall_curve.png
│   │       ├── IoU_distribution.png
│   │       └── sample_detections.png
│   │
│   └── run_20260421_101523/       ← Second run (new timestamp)
│       └── ... (same structure)
│
├── exp02_detection_modified_v1/
│   └── run_TIMESTAMP/
│       ├── model/
│       ├── logs/
│       └── figures/
│
... (exp03-exp06 follow same pattern)
```

### 6.2 Output Contents

#### Model Folder
- `best_model.pt` / `best_model.pth`: Trained model weights
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```
  epoch, train_loss, val_loss, train_metric, val_metric, learning_rate
  1, 2.345, 2.123, 0.45, 0.48, 0.001
  2, 1.987, 1.876, 0.52, 0.55, 0.001
  ...
  ```
- `experiment_report.md`: Comprehensive markdown report including:
  - Experiment configuration
  - Training summary
  - Evaluation metrics
  - Figure references
  - Hardware info
  - Execution time

#### Figures Folder
- **Detection experiments**:
  - `precision_recall_curve.png`: PR curve for each class
  - `IoU_distribution.png`: Histogram of IoU values
  - `sample_detections.png`: Example predictions on test images
  - `confusion_matrix.png`: If multi-class detection
  
- **Classification experiments**:
  - `confusion_matrix.png`: 5×5 confusion matrix
  - `roc_curve.png`: ROC curves (one-vs-rest for each class)
  - `per_class_metrics.png`: Bar chart of precision/recall/F1 per class
  - `training_curves.png`: Loss and accuracy over epochs

### 6.3 Example Experiment Report (Markdown)

```markdown
# Experiment Report: exp01_detection_baseline

## Experiment Information
- **Date**: 2026-04-20 19:30:45
- **Duration**: 2h 15m 32s
- **GPU**: NVIDIA T4 (16GB)
- **PyTorch Version**: 2.0.1

## Model Configuration
```json
{
  "backbone": "m",
  "input_size": 640,
  "confidence_threshold": 0.5,
  "nms_iou_threshold": 0.45
}
```

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 50
- **Early Stopping Patience**: 10
- **Mixed Precision**: Enabled
- **Gradient Accumulation**: 1 step

## Training Summary
- **Total Epochs Completed**: 42 (early stopped at epoch 42)
- **Best Validation mAP@0.5**: 0.78 (epoch 38)
- **Final Training Loss**: 1.234
- **Final Validation Loss**: 1.456

## Evaluation Metrics (Test Set)
- **mAP@0.5**: 0.76
- **mAP@0.5:0.95**: 0.54
- **Average IoU**: 0.68
- **Precision**: 0.82
- **Recall**: 0.74

## Figures
- ![Precision-Recall Curve](figures/precision_recall_curve.png)
- ![IoU Distribution](figures/IoU_distribution.png)
- ![Sample Detections](figures/sample_detections.png)

## Observations
- Model converged quickly in first 20 epochs
- Early stopping triggered at epoch 42
- Best performance at epoch 38
- Some false positives on small/occluded faces

## Recommendations
- Try larger input size (1280) for better small object detection
- Increase training data augmentation for occluded faces
- Consider adjusting confidence threshold to 0.6 for fewer false positives
```

---

## 7. Key Design Principles

### 7.1 Simplicity
- **One model class per task**: YOLOv8Detector and ResNet50Classifier
- **Configuration-driven variants**: Different experiments use same class with different configs
- **Simplified output structure**: Only 6 top-level folders, timestamped runs

### 7.2 Reusability
- **Data preprocessing once**: Processed data reused across all experiments
- **Modular design**: Separate modules for data, models, training, evaluation
- **Template-based experiments**: Easy to create new experiments from template

### 7.3 Reproducibility
- **Timestamped runs**: Each execution creates unique output folder
- **Config logging**: All hyperparameters saved to JSON
- **Comprehensive reports**: Markdown reports with all details
- **Version control friendly**: Clear separation of code and outputs

### 7.4 Scalability
- **Easy to add experiments**: Copy template, modify config
- **Easy to compare runs**: Same experiment, different timestamps
- **Extensible architecture**: Add new models by creating new wrapper classes

### 7.5 Performance Optimization
- **Mixed precision training**: AMP for memory efficiency
- **Gradient accumulation**: Simulate larger batch sizes
- **Two-phase training**: Freeze backbone initially
- **Progress monitoring**: tqdm progress bars for real-time feedback

### 7.6 Error Handling
- **Data validation**: Check dataset integrity before training
- **OOM protection**: Automatic batch size reduction
- **Graceful degradation**: CPU fallback if GPU unavailable
- **Checkpoint recovery**: Resume from last checkpoint if interrupted

---

## 8. Quick Start Guide

### 8.1 Installation
```bash
# Clone repository
git clone <repo_url>
cd CNN_A3

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Download Data
```bash
# Run once - downloads both datasets
python src/data_processing/download_datasets.py
```

### 8.3 Preprocess Data
```bash
# Automatically called by experiments, or run manually
python -c "from src.data_processing.detection_preprocessor import DetectionPreprocessor; DetectionPreprocessor().process()"
python -c "from src.data_processing.emotion_preprocessor import EmotionPreprocessor; EmotionPreprocessor().process()"
```

### 8.4 Run Experiments
```bash
# Run single experiment
cd experiments
python exp01_detection_baseline.py

# Run all experiments
bash ../scripts/run_all_experiments.sh
```

### 8.5 View Results
```bash
# Check output directory
ls outputs/exp01_detection_baseline/

# View latest run
ls outputs/exp01_detection_baseline/run_*/

# Open report
cat outputs/exp01_detection_baseline/run_*/logs/experiment_report.md
```

### 8.6 Run Inference
```python
# Detection only
from src.inference.detection_inference import DetectionInference
inf = DetectionInference('outputs/exp01_detection_baseline/run_*/model/best_model.pt')
results = inf.predict('test_image.jpg')

# Classification only
from src.inference.classification_inference import ClassificationInference
inf = ClassificationInference('outputs/exp04_classification_baseline/run_*/model/best_model.pth')
results = inf.predict('cropped_face.jpg')

# End-to-end pipeline
from src.inference.pipeline_inference import PipelineInference
pipeline = PipelineInference(
    'outputs/exp01_detection_baseline/run_*/model/best_model.pt',
    'outputs/exp04_classification_baseline/run_*/model/best_model.pth'
)
results = pipeline.predict('test_image.jpg')
```

---

## 9. Troubleshooting

### Common Issues

#### Issue 1: Out of Memory (OOM)
**Solution**:
```python
# Reduce batch size
training_config['batch_size'] = 8

# Enable gradient accumulation
training_config['gradient_accumulation_steps'] = 4

# Enable mixed precision
training_config['use_amp'] = True
```

#### Issue 2: Slow Training
**Solution**:
- Ensure GPU is being used: `torch.cuda.is_available()`
- Use mixed precision: `use_amp=True`
- Reduce number of epochs for testing
- Use smaller model variant (backbone='s' instead of 'm')

#### Issue 3: Poor Model Performance
**Solution**:
- Check data quality and annotations
- Increase data augmentation
- Adjust learning rate (try 1e-4 or 1e-5)
- Train for more epochs
- Try different model variant

#### Issue 4: Missing Dependencies
**Solution**:
```bash
pip install -r requirements.txt
# Or install individually
pip install torch torchvision ultralytics opencv-python matplotlib pandas
```

---

## 10. Future Enhancements

### Potential Improvements
1. **Model Ensemble**: Combine multiple models for better accuracy
2. **Active Learning**: Identify hard examples for manual labeling
3. **Model Quantization**: INT8 quantization for faster inference
4. **ONNX Export**: Export models for deployment
5. **Web Interface**: Gradio/Streamlit demo application
6. **Video Processing**: Real-time video stream processing
7. **Data Versioning**: DVC for dataset version control
8. **Experiment Tracking**: Weights & Biases or MLflow integration

---

## 11. References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Dog Face Detection Dataset](https://www.kaggle.com/datasets/jessicali9530/dog-face-detection)
- [Dog Emotion Dataset](https://www.kaggle.com/datasets/tongpython/dog-emotions-5-classes)

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-20  
**Author**: CNN_A3 Project Team
