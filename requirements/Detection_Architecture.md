# Detection Task Architecture - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-05-04  

---

## Architecture Overview

The detection task implements two mainstream object detection frameworks with modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS (Flow Control)               │
│                                                               │
│  YOLOv8 Series:                                              │
│  ├── detection_YOLOv8_v1.py       (Baseline)                 │
│  ├── detection_YOLOv8_v2.py       (Deeper Backbone)          │
│  └── detection_YOLOv8_v3.py       (Shallower Backbone)       │
│                                                               │
│  Faster R-CNN Series:                                        │
│  └── exp02_detection_FasterRCNN.py  (Template)              │
│                                                               │
│  Responsibilities:                                            │
│  - Load dataset config                                       │
│  - Initialize model & parameters                             │
│  - Orchestrate training                                      │
│  - Call evaluator                                            │
│  - Save results to output/                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│  MODELS  │ │ TRAINING │ │ EVALUATION   │
│          │ │          │ │              │
│ Provides │ │ Handles  │ │ Calculates   │
│ model    │ │ training │ │ metrics &    │
│ classes  │ │ loop     │ │ generates    │
│ & config │ │          │ │ reports      │
└──────────┘ └──────────┘ └──────────────┘
```

---

## Detection Frameworks

### 1. YOLOv8 Series (Single-Stage Detector)

**Architecture Type:** Anchor-free, single-stage detector  
**Implementation:** Ultralytics YOLOv8 framework  
**Customization Approach:** Direct PyTorch-based backbone modification

**Key Features:**
- Fast inference speed suitable for real-time applications
- Supports multiple model scales (n/s/m/l/x)
- Built-in data augmentation and training optimizations
- Easy customization through direct layer manipulation

**Customization Strategy:**
- **V1 (Baseline):** Standard YOLOv8m without modifications
- **V2 (Deeper):** Adds convolutional layers to backbone for enhanced feature extraction
- **V3 (Shallower):** Reduces C2f module repeats for lighter, faster model

**See detailed implementation:** [YOLOv8_Experiments_Summary.md](./YOLOv8_Experiments_Summary.md)

---

### 2. Faster R-CNN Series (Two-Stage Detector)

**Architecture Type:** Region-based, two-stage detector  
**Implementation:** Torchvision Faster R-CNN with ResNet50+FPN backbone  
**Status:** Template created, awaiting dataloader implementation

**Key Features:**
- Higher accuracy potential through region proposal mechanism
- Slower inference compared to single-stage detectors
- More complex training pipeline requiring custom dataloaders
- Better suited for scenarios prioritizing accuracy over speed

**Planned Customization:**
- Modify backbone depth/width
- Adjust FPN configuration
- Experiment with different anchor strategies

---

## Module Responsibilities

### 1. `experiments/` - Flow Control

**Responsibilities:**
- Load dataset configuration from YAML files
- Initialize detection models with appropriate configurations
- Configure training hyperparameters
- Orchestrate training → evaluation → summary generation pipeline
- Manage output directory structure with timestamps

**Design Principle:** Experiments contain only flow control logic, no implementation details.

---

### 2. `src/models/` - Model Definitions

**Files:**
- `YOLOv8DetectorModel.py` - YOLOv8 wrapper with PyTorch-based customization
- `FasterRCNNDetectorModel.py` - Faster R-CNN wrapper

**YOLOv8 Customization Implementation:**
- **Direct Layer Manipulation:** Modifies backbone using PyTorch operations
- **No YAML Dependencies:** Avoids index management issues in YAML configs
- **Dynamic Modification:** `_add_conv_layers()` and `_reduce_conv_layers()` methods
- **Pretrained Weight Compatibility:** Loads standard weights then applies modifications

**Configuration Pattern:**
```python
YOLOV8_BASELINE_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'pretrained': True,
    'customize_type': None  # No customization
}

YOLOV8_V2_CONFIG = {
    'backbone': 'm',
    'customize_type': 'deeper'  # Adds conv layers
}

YOLOV8_V3_CONFIG = {
    'backbone': 'm',
    'customize_type': 'shallower'  # Reduces conv layers
}
```

---

### 3. `src/training/` - Training Frameworks

**Files:**
- `YOLOv8_trainer.py` - YOLOv8 trainer using Ultralytics API
- `FasterRCNN_trainer.py` - Faster R-CNN trainer (custom loop)

**YOLOv8 Training Features:**
- Centralized configuration via `YOLOV8_V[1-3]_CONFIG` dictionaries
- Automatic mixed precision (AMP) support
- Configurable optimizer, learning rate schedules, early stopping
- Integrated with Ultralytics training pipeline
- CSV metrics logging for detailed analysis

**Training Configuration Example:**
```python
YOLOV8_V2_CONFIG = {
    'learning_rate': 0.0005,
    'batch_size': 12,
    'epochs': 120,
    'optimizer': 'adam',
    'weight_decay': 5e-4,
    'use_amp': True,
    'patience': 20,
    'cos_lr': True,
    'close_mosaic': 10
}
```

---

### 4. `src/evaluation/` - Evaluation Framework

**File:** `detection_evaluator.py`

**Capabilities:**
- **YOLOv8 Evaluation:** Leverages Ultralytics built-in validation
- **Metrics Calculation:** mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Result Visualization:** Confusion matrices, PR curves (future enhancement)
- **Summary Generation:** Automated Markdown report creation

**Output Structure:**
```
outputs/detection_yolov8_v[X]/run_TIMESTAMP/
├── training/
│   ├── train/
│   │   ├── weights/best.pt
│   │   └── results.csv
│   └── training_history.csv
├── evaluation/
│   └── evaluation_metrics.json
└── experiment_summary.md
```

---

## Key Design Principles

### 1. Separation of Concerns
- **Experiments:** Flow orchestration only
- **Models:** Architecture definition + customization logic
- **Training:** Hyperparameter management + training loop
- **Evaluation:** Metrics calculation + reporting

### 2. Modular Customization
- YOLOv8 uses PyTorch-based direct layer manipulation
- Avoids YAML index management complexity
- Enables dynamic architecture changes at runtime
- Maintains pretrained weight compatibility

### 3. Configuration Centralization
- All hyperparameters in dedicated config dictionaries
- Clear mapping between experiments and configurations
- Easy comparison across variants
- Reproducible experimental setups

### 4. Comprehensive Tracking
- CSV logging for epoch-by-epoch metrics
- Automated Markdown summaries
- Timestamped output directories
- Complete configuration documentation

---

## Dataset Configuration

**Dataset Location:** `data/25509225/Object_Detection/yolo/data.yaml`

**Format:** Ultralytics YOLO format with:
- Train/valid/test split paths
- Class names and count
- Pre-segregated datasets (no resplitting required)

**Compliance:** Uses provided splits as per teacher requirements.

---

## Hardware Optimization

**Target GPU:** NVIDIA T4 (16GB VRAM, ~10GB usable)

**Batch Size Recommendations:**
- YOLOv8 Baseline (V1): 16
- YOLOv8 Deeper (V2): 12 (reduced due to larger model)
- YOLOv8 Shallower (V3): 20 (increased due to smaller model)
- Faster R-CNN: 2-4 (two-stage detector memory intensive)

**Optimization Techniques:**
- Mixed precision training (AMP) enabled by default
- Appropriate `num_workers` for DataLoader
- Early stopping to prevent unnecessary training

---

## Comparison Matrix

| Aspect | YOLOv8 Series | Faster R-CNN Series |
|--------|--------------|---------------------|
| **Detector Type** | Single-stage | Two-stage |
| **Speed** | Fast (real-time capable) | Slower (accuracy-focused) |
| **Implementation** | Ultralytics framework | Torchvision + custom loop |
| **Customization** | PyTorch-based layer manipulation | Planned: backbone/FPN mods |
| **Training Complexity** | Low (single command) | High (custom dataloader needed) |
| **Memory Usage** | Moderate | High |
| **Best For** | Speed-critical applications | Accuracy-critical scenarios |

---

## Running Experiments

### YOLOv8 Series (Fully Implemented)

```bash
# V1: Baseline
python experiments/detection_YOLOv8_v1.py

# V2: Deeper Backbone
python experiments/detection_YOLOv8_v2.py

# V3: Shallower Backbone
python experiments/detection_YOLOv8_v3.py
```

### Faster R-CNN (Template Stage)

```bash
python experiments/exp02_detection_FasterRCNN.py
```

**Note:** Faster R-CNN requires dataloader implementation before full execution.

---

## Documentation Structure

This document provides high-level architectural overview. For detailed implementation specifics:

- **YOLOv8 Details:** See [YOLOv8_Experiments_Summary.md](./YOLOv8_Experiments_Summary.md)
  - Exact layer modifications
  - Training hyperparameters
  - Performance comparisons
  - Analysis and findings

- **Faster R-CNN Details:** See [FasterRCNN_DataLoader.md](./FasterRCNN_DataLoader.md)
  - Dataloader implementation guide
  - Training pipeline details
  - Evaluation strategy

---

## Next Steps

### Immediate Priorities
1. ✅ YOLOv8 V1/V2/V3 fully implemented and tested
2. ⏳ Run all three YOLOv8 experiments
3. ⏳ Analyze results and generate comparison plots
4. ⚠️ Implement Faster R-CNN dataloader
5. ⚠️ Complete Faster R-CNN training pipeline

### Future Enhancements
- Add visualization tools for detection results
- Implement cross-framework comparison utilities
- Extend to additional detection architectures (SSD, RetinaNet)

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Last Updated:** 2026-05-04
