# Classification Task Architecture - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-04-30  

---

## Architecture Overview

The classification task follows a clean, modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS (Flow Control)               │
│  exp03_classification_ResNet50_v1.py  (Baseline)            │
│  exp04_classification_ResNet50_v2.py  (Customized)          │
│                                                               │
│  Responsibilities:                                            │
│  - Load data (DataLoaders)                                   │
│  - Initialize model & parameters                             │
│  - Orchestrate training phases                               │
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
- `exp03_classification_ResNet50_v1.py` - Baseline experiment
- `exp04_classification_ResNet50_v2.py` - Customized experiment

**Responsibilities:**
- Define data loading pipeline (transforms, DataLoaders)
- Initialize model with configuration
- Orchestrate multi-phase training
- Call trainer and evaluator
- Save results and generate summary

**Example Flow:**
```python
# 1. Load data
train_loader, val_loader, test_loader, class_names = create_dataloaders(...)

# 2. Initialize model
model = ResNet50Classifier(**BASELINE_CONFIG)
model.freeze_backbone()

# 3. Train Phase 1
trainer1 = ClassificationTrainer(model, lr=1e-3)
trainer1.train(train_loader, val_loader, criterion, epochs=15, ...)

# 4. Train Phase 2
model.unfreeze_backbone()
trainer2 = ClassificationTrainer(model, lr=1e-4)
trainer2.train(train_loader, val_loader, criterion, epochs=35, ...)

# 5. Evaluate
evaluator = ClassificationEvaluator(class_names)
metrics = evaluator.evaluate(model, test_loader, output_dir)

# 6. Save summary
save_experiment_summary(...)
```

---

### 2. `src/models/` - Model Definitions

**File:** `ResNet50ClassifierModel.py`

**Provides:**
- `ResNet50Classifier` class
- Configuration dictionaries (`BASELINE_CONFIG`, `CUSTOMIZED_CONFIG`)

**Responsibilities:**
- Define model architecture
- Provide forward pass
- Support freeze/unfreeze operations

**Configurations:**

```python
# Baseline (exp03)
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,  # Single FC layer
    'use_batch_norm': True
}

# Customized (exp04)
CUSTOMIZED_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.7,             # Higher dropout
    'pretrained': True,
    'additional_fc_layers': True,    # Multi-layer FC with BatchNorm
    'use_batch_norm': True
}
```

**Architecture Differences:**

| Component | Baseline | Customized |
|-----------|----------|------------|
| Classifier Head | 2048 → 10 | 2048 → 512 → 256 → 10 |
| BatchNorm | No | Yes (in FC layers) |
| Dropout | 0.5 | 0.7 |

---

### 3. `src/training/` - Training Framework

**File:** `classification_trainer.py`

**Provides:** `ClassificationTrainer` class

**Responsibilities:**
- Training loop (forward/backward pass)
- Validation loop
- Optimization (AdamW)
- Mixed precision training (AMP)
- Model checkpointing (best model)
- Early stopping

**Key Methods:**
- `train_epoch()` - One epoch of training
- `validate()` - Validation on val set
- `train()` - Full training loop with early stopping

**Usage:**
```python
trainer = ClassificationTrainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-4,
    use_amp=True
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    epochs=50,
    output_dir='outputs/phase1',
    patience=10
)
```

---

### 4. `src/evaluation/` - Evaluation Framework

**File:** `classification_evaluator.py`

**Provides:** `ClassificationEvaluator` class

**Responsibilities:**
- Generate predictions on test set
- Calculate metrics (accuracy, precision, recall, F1)
- Create confusion matrix
- Save evaluation results (JSON + report)

**Metrics Calculated:**
- Overall: accuracy, precision, recall, F1 (weighted & macro)
- Per-class: precision, recall, F1
- Confusion matrix

**Usage:**
```python
evaluator = ClassificationEvaluator(class_names=class_names)

metrics = evaluator.evaluate(
    model=model,
    test_loader=test_loader,
    output_dir='outputs/evaluation'
)

# metrics contains:
# - accuracy
# - precision_weighted, recall_weighted, f1_weighted
# - precision_macro, recall_macro, f1_macro
# - per_class metrics
# - confusion_matrix
```

---

## Experiment Comparison

### Experiment 03 (Baseline)

**Configuration:**
- Model: ResNet50 with single FC layer
- Dropout: 0.5
- Augmentation: Standard (flip, rotation, color jitter)
- Fine-tuning: Unfreeze layer3+layer4 only
- Weight decay: 1e-4
- Label smoothing: 0.1
- Epochs: 15 (frozen) + 35 (fine-tune) = 50 total

### Experiment 04 (Customized)

**Modifications:**
1. **Architecture:** Additional FC layers (2048→512→256→10) with BatchNorm
2. **Regularization:** Higher dropout (0.7), higher weight decay (5e-3)
3. **Augmentation:** Enhanced (stronger color jitter, random affine)
4. **Fine-tuning:** Extended (unfreeze layer2+layer3+layer4)
5. **Label smoothing:** Increased to 0.15
6. **Epochs:** 15 (frozen) + 45 (fine-tune) = 60 total

**Hypothesis:**
- Additional layers enable learning complex feature combinations
- Extended unfreezing adapts lower-level features to bird-specific patterns
- Stronger regularization prevents overfitting despite increased capacity

---

## File Structure

```
CNN_A2/
├── experiments/                          [Flow Control]
│   ├── exp03_classification_ResNet50_v1.py   (Baseline)
│   └── exp04_classification_ResNet50_v2.py   (Customized)
│
├── src/
│   ├── models/                         [Model Definitions]
│   │   ├── __init__.py
│   │   └── ResNet50ClassifierModel.py      (ResNet50 + configs)
│   │
│   ├── training/                       [Training Framework]
│   │   ├── __init__.py
│   │   └── classification_trainer.py       (Training loop)
│   │
│   └── evaluation/                     [Evaluation Framework]
│       ├── __init__.py
│       └── classification_evaluator.py     (Metrics & reports)
│
├── data/25509225/Image_Classification/split_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
└── outputs/                            [Results]
    ├── exp03_baseline_TIMESTAMP/
    │   ├── phase1/
    │   │   └── best_model.pth
    │   ├── phase2/
    │   │   └── best_model.pth
    │   ├── evaluation/
    │   │   ├── evaluation_metrics.json
    │   │   └── classification_report.txt
    │   └── experiment_summary.md
    │
    └── exp04_customized_TIMESTAMP/
        └── ... (same structure)
```

---

## Running Experiments

### Quick Start

```bash
# Run baseline experiment
python experiments/exp03_classification_ResNet50_v1.py

# Run customized experiment
python experiments/exp04_classification_ResNet50_v2.py
```

### Expected Output

Each experiment creates a timestamped directory in `outputs/` containing:
- Trained models (phase1/best_model.pth, phase2/best_model.pth)
- Evaluation metrics (evaluation/evaluation_metrics.json)
- Experiment summary (experiment_summary.md)

---

## Key Design Principles

1. **Separation of Concerns:**
   - Experiments control the flow
   - Models define architecture
   - Training handles optimization
   - Evaluation calculates metrics

2. **Simplicity:**
   - Each module has a single responsibility
   - Minimal abstraction
   - Clear interfaces

3. **Reproducibility:**
   - Fixed random seeds (student ID)
   - Deterministic data splitting
   - Saved configurations

4. **Comparability:**
   - Same data loaders for both experiments
   - Consistent evaluation metrics
   - Side-by-side result comparison

---

## Next Steps

1. Run both experiments
2. Compare results in `outputs/` directories
3. Analyze `experiment_summary.md` files
4. Include comparison in assignment report

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks
