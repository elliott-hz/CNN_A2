# Detection Task Architecture - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-04-30  

---

## Architecture Overview

The detection task follows the same clean, modular architecture as classification:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS (Flow Control)               │
│  exp01_detection_YOLOv8.py          (YOLOv8 Baseline)       │
│  exp02_detection_FasterRCNN.py      (Faster R-CNN Template) │
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

## Module Responsibilities

### 1. `experiments/` - Flow Control

**Files:**
- `exp01_detection_YOLOv8.py` - YOLOv8 baseline experiment
- `exp02_detection_FasterRCNN.py` - Faster R-CNN experiment (template)

**Responsibilities:**
- Load dataset configuration
- Initialize model with configuration
- Run training
- Call evaluator
- Save results and generate summary

**Example Flow (YOLOv8):**
```python
# 1. Load dataset config
with open('data/processed/detection/dataset.yaml') as f:
    dataset_config = yaml.safe_load(f)

# 2. Initialize model
model = YOLOv8Detector(**YOLOV8_BASELINE_CONFIG)

# 3. Train
trainer = YOLOv8Trainer(lr=0.001, batch_size=24, epochs=100)
results = trainer.train(model, train_data, val_data, output_dir)

# 4. Evaluate
evaluator = DetectionEvaluator()
metrics = evaluator.evaluate_yolov8(model, test_data, output_dir)

# 5. Save summary
save_experiment_summary(...)
```

---

### 2. `src/models/` - Model Definitions

**Files:**
- `YOLOv8DetectorModel.py` - YOLOv8 wrapper
- `FasterRCNNDetectorModel.py` - Faster R-CNN wrapper

**Provides:**
- Model classes
- Configuration dictionaries

#### YOLOv8 Configuration

```python
YOLOV8_BASELINE_CONFIG = {
    'backbone': 'm',                  # Medium model
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True
}
```

#### Faster R-CNN Configuration

```python
FASTERRCNN_BASELINE_CONFIG = {
    'num_classes': 2,                 # 1 class + background
    'pretrained': True,
    'min_size': 640,
    'max_size': 640
}
```

---

### 3. `src/training/` - Training Frameworks

**Files:**
- `YOLOv8_trainer.py` - YOLOv8 trainer (uses Ultralytics)
- `FasterRCNN_trainer.py` - Faster R-CNN trainer (custom loop)

#### YOLOv8 Trainer

**Responsibilities:**
- Configure Ultralytics training
- Run training via `model.train()`
- Save results

**Usage:**
```python
trainer = YOLOv8Trainer(
    learning_rate=0.001,
    batch_size=24,
    epochs=100,
    optimizer='adam',
    weight_decay=1e-4,
    use_amp=True
)

results = trainer.train(
    model=model,
    train_data='dataset.yaml',
    val_data='dataset.yaml',
    output_dir='outputs/training'
)
```

#### Faster R-CNN Trainer

**Responsibilities:**
- Custom training loop
- Optimizer management
- Model checkpointing
- Early stopping

**Note:** Requires custom dataloader implementation for your dataset format.

---

### 4. `src/evaluation/` - Evaluation Framework

**File:** `detection_evaluator.py`

**Provides:** `DetectionEvaluator` class

**Methods:**
- `evaluate_yolov8()` - Evaluate YOLOv8 using Ultralytics validation
- `evaluate_fasterrcnn()` - Evaluate Faster R-CNN (placeholder)

**Metrics:**
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

**Usage:**
```python
evaluator = DetectionEvaluator()

# For YOLOv8
metrics = evaluator.evaluate_yolov8(
    model=model,
    test_dataset='dataset.yaml',
    output_dir='outputs/evaluation'
)

# For Faster R-CNN (needs implementation)
metrics = evaluator.evaluate_fasterrcnn(
    model=model,
    test_loader=test_loader,
    output_dir='outputs/evaluation'
)
```

---

## Experiment Comparison

### Experiment 01: YOLOv8 Baseline

**Configuration:**
- Model: YOLOv8m (medium)
- Input size: 640x640
- Confidence threshold: 0.5
- NMS IoU: 0.45
- Epochs: 100
- Batch size: 24
- Optimizer: Adam
- Learning rate: 0.001
- Mixed precision: Enabled

**Status:** ✅ Fully implemented and ready to run

### Experiment 02: Faster R-CNN Baseline

**Configuration:**
- Model: Faster R-CNN with ResNet50+FPN
- Number of classes: 2 (1 class + background)
- Image size: 640x640
- Epochs: 50 (planned)
- Batch size: 4 (smaller due to memory)
- Learning rate: 0.001

**Status:** ⚠️ Template created, needs dataloader implementation

**TODO:**
1. Implement data loader for Faster R-CNN format
2. Complete training loop
3. Implement evaluation metrics calculation

---

## File Structure

```
CNN_A2/
├── experiments/                          [Flow Control]
│   ├── exp01_detection_YOLOv8.py             (✅ Ready)
│   └── exp02_detection_FasterRCNN.py         (⚠️ Template)
│
├── src/
│   ├── models/                         [Model Definitions]
│   │   ├── __init__.py
│   │   ├── ResNet50ClassifierModel.py      (Classification)
│   │   ├── YOLOv8DetectorModel.py          (Detection)
│   │   └── FasterRCNNDetectorModel.py      (Detection)
│   │
│   ├── training/                       [Training Frameworks]
│   │   ├── __init__.py
│   │   ├── classification_trainer.py       (Classification)
│   │   ├── YOLOv8_trainer.py               (Detection)
│   │   └── FasterRCNN_trainer.py           (Detection)
│   │
│   └── evaluation/                     [Evaluation Frameworks]
│       ├── __init__.py
│       ├── classification_evaluator.py     (Classification)
│       └── detection_evaluator.py          (Detection)
│
├── data/processed/detection/
│   └── dataset.yaml                    [Dataset Config]
│
└── outputs/                            [Results]
    ├── exp01_yolov8_TIMESTAMP/
    │   ├── training/
    │   │   └── ... (Ultralytics outputs)
    │   ├── evaluation/
    │   │   └── evaluation_metrics.json
    │   └── experiment_summary.md
    │
    └── exp02_fasterrcnn_TIMESTAMP/
        └── ... (same structure)
```

---

## Running Experiments

### YOLOv8 (Ready to Run)

```bash
python experiments/exp01_detection_YOLOv8.py
```

**Prerequisites:**
- Dataset preprocessed: `bash scripts/run_data_preprocessing.sh`
- Dataset config exists: `data/processed/detection/dataset.yaml`

### Faster R-CNN (Template Only)

```bash
python experiments/exp02_detection_FasterRCNN.py
```

**Current Status:** Creates output directory and summary, but skips training/evaluation.

**To Complete:**
1. Implement `create_dataloaders()` function
2. Uncomment training code in `main()`
3. Implement Faster R-CNN evaluation in `DetectionEvaluator`

---

## Key Design Principles

1. **Separation of Concerns:**
   - Experiments control the flow
   - Models define architecture
   - Training handles optimization
   - Evaluation calculates metrics

2. **Modularity:**
   - YOLOv8 and Faster R-CNN have separate implementations
   - No interference with classification code
   - Easy to add new detection models (SSD, etc.)

3. **Simplicity:**
   - Each module has a single responsibility
   - Minimal abstraction
   - Clear interfaces

4. **Reproducibility:**
   - Fixed configurations
   - Saved experiment summaries
   - Timestamped output directories

---

## Next Steps

### For YOLOv8:
1. Ensure dataset is preprocessed
2. Run experiment: `python experiments/exp01_detection_YOLOv8.py`
3. Monitor training progress
4. Analyze results in `outputs/`

### For Faster R-CNN:
1. Implement dataloader for your dataset format
2. Complete training loop in `FasterRCNNTrainer`
3. Implement evaluation metrics
4. Test and validate

---

## Comparison with Classification

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| Models | ResNet50 only | YOLOv8, Faster R-CNN |
| Data Format | ImageFolder | COCO/YOLO format |
| Training | Custom loop (PyTorch) | Ultralytics (YOLO) / Custom (Faster R-CNN) |
| Metrics | Accuracy, F1, etc. | mAP, Precision, Recall |
| Complexity | Simpler | More complex (bounding boxes) |

Both follow the same architectural pattern:
- Experiments → Flow control
- Models → Architecture definition
- Training → Optimization
- Evaluation → Metrics

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks
