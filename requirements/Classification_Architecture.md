# Classification Task Architecture - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-05-04  

---

## Architecture Overview

The classification task follows a clean, modular architecture with clear separation of concerns and centralized configuration management:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS (Flow Control)               │
│                                                               │
│  classification_ResNet50_baseline.py           (Baseline)    │
│  classification_ResNet50_baseline_gridsearch.py (GridSearch) │
│                                                               │
│  FC Enhancement:                                              │
│  ├── classification_ResNet50_FC_v1.py         (FC v1)        │
│                                                               │
│  Reduced Depth Models:                                        │
│  ├── classification_ResNet50_reduced_v1.py    (Reduced v1)   │
│  └── classification_ResNet50_reduced_v2.py    (Reduced v2)   │
│                                                               │
│  Deeper Backbone Models:                                      │
│  ├── classification_ResNet50_deeper_v1.py     (Deeper v1)    │
│  ├── classification_ResNet50_deeper_v2.py     (Deeper v2)    │
│  └── classification_ResNet50_deeper_v3.py     (Deeper v3)    │
│                                                               │
│  Responsibilities:                                            │
│  - Load data via unified ClassificationDataLoader            │
│  - Initialize model with configuration                       │
│  - Train with TrainingConfig                                 │
│  - Evaluate and generate summary                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┼───────────┬───────────────┐
       │           │           │               │
       ▼           ▼           ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  MODELS  │ │ TRAINING │ │   DATA   │ │ EVALUATION   │
│          │ │          │ │ LOADING  │ │              │
│ Provides │ │ Handles  │ │ Unified  │ │ Calculates   │
│ model    │ │ training │ │ loader   │ │ metrics &    │
│ classes  │ │ loop     │ │ module   │ │ generates    │
│ & config │ │          │ │          │ │ reports      │
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

---

## Module Responsibilities

### 1. `experiments/` - Flow Control

**Files:**

**Baseline Experiments:**
- `classification_ResNet50_baseline.py` - Baseline experiment (standard ResNet50)
- `classification_ResNet50_baseline_gridsearch.py` - Hyperparameter grid search for baseline optimization

**FC Enhancement:**
- `classification_ResNet50_FC_v1.py` - Enhanced multi-layer FC head (NOT true CNN customization)

**Reduced Depth Models (TRUE CNN):**
- `classification_ResNet50_reduced_v1.py` - Removed layer3 to reduce depth
- `classification_ResNet50_reduced_v2.py` - Removed layer4 to reduce depth further

**Deeper Backbone Models (TRUE CNN):**
- `classification_ResNet50_deeper_v1.py` - Added conv blocks after layer1
- `classification_ResNet50_deeper_v2.py` - Added conv blocks after layer2
- `classification_ResNet50_deeper_v3.py` - Added conv blocks after layer3

**Responsibilities:**
- Import unified data loader from `ClassificationDataLoader`
- Initialize model with model configuration
- Create trainer with training configuration
- Orchestrate single-phase training (NO freezing)
- Call evaluator for metrics and visualization
- Generate comprehensive experiment summary

**Example Flow:**
```python
from src.data_processing.ClassificationDataLoader import create_baseline_dataloaders
from src.models.ResNet50ClassifierModel import ResNet50Classifier, BASELINE_CONFIG
from src.training.ResNet50_trainer import ClassificationTrainer, TRAINING_CONFIG_BASELINE
from src.evaluation.classification_evaluator import ClassificationEvaluator

# 1. Load data (unified loader)
train_loader, val_loader, test_loader, class_names = create_baseline_dataloaders(DATA_ROOT)

# 2. Initialize model (all layers trainable)
model = ResNet50Classifier(**BASELINE_CONFIG)

# 3. Train (single phase, no freezing)
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_BASELINE.label_smoothing)
history = trainer.train(train_loader, val_loader, criterion, output_dir)

# 4. Evaluate
evaluator = ClassificationEvaluator(class_names)
metrics = evaluator.evaluate(model, test_loader, output_dir)

# 5. Visualize and analyze
evaluator.plot_training_curves(history['history'], output_dir)
analysis = evaluator.analyze_overfitting(history['history'])

# 6. Generate summary (abstracted in evaluator)
evaluator.generate_experiment_summary(
    experiment_name='baseline',
    model_config=BASELINE_CONFIG,
    training_config=TRAINING_CONFIG_BASELINE,
    trainer_metrics={'best_val_acc': trainer.best_val_acc},
    evaluation_metrics=metrics,
    overfitting_analysis=analysis,
    output_dir=output_dir
)
```

**Key Changes from Previous Version:**
- ❌ **Removed:** Two-phase training with layer freezing
- ✅ **Added:** Single-phase training with all layers trainable
- ✅ **Added:** Unified data loading module
- ✅ **Added:** Centralized training configuration
- ✅ **Added:** Automated summary generation in evaluator

---

### 2. `src/models/` - Model Definitions

**File:** `ResNet50ClassifierModel.py`

**Provides:**
- `ResNet50Classifier` class with backbone modification support
- Nine configuration dictionaries for different experiments

**Responsibilities:**
- Define model architecture (standard or customized)
- Support true CNN customization (backbone modification)
- Provide forward pass
- All layers trainable by default (no freezing methods)

**Configurations:**

```python
# Baseline (classification_ResNet50_baseline.py)
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': False             # Standard ResNet50
}

# FC v1 (classification_ResNet50_FC_v1.py) - Enhanced FC Head
CUSTOMIZED_V1_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.7,                # Higher dropout
    'pretrained': True,
    'additional_fc_layers': True,       # Multi-layer FC with BatchNorm
    'use_batch_norm': True,
    'modify_backbone': False            # Standard backbone
}

# Reduced v1 (classification_ResNet50_reduced_v1.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V3_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': 'layer3',           # Remove layer3 (reduce depth)
    'add_conv_after_layer': None
}

# Reduced v2 (classification_ResNet50_reduced_v2.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V4_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': 'layer4',           # Remove layer4 (reduce depth further)
    'add_conv_after_layer': None
}

# Deeper v1 (classification_ResNet50_deeper_v1.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V5_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': None,
    'add_conv_after_layer': 'layer1'    # Add conv blocks after layer1
}

# Deeper v2 (classification_ResNet50_deeper_v2.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V6_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': None,
    'add_conv_after_layer': 'layer2'    # Add conv blocks after layer2
}

# Deeper v3 (classification_ResNet50_deeper_v3.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V7_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,      # Single FC layer
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': None,
    'add_conv_after_layer': 'layer3'    # Add conv blocks after layer3
}
```

**Architecture Differences:**

| Experiment | Backbone Modification | FC Head | Dropout | Params | Customization Type |
|------------|----------------------|---------|---------|--------|-------------------|
| **Baseline** | Standard | 2048 → 10 | 0.5 | ~25.6M | None (control) |
| **FC_v1** | Standard | 2048→512→256→10 + BN | 0.7 | ~25.6M | FC only |
| **Reduced_v1** | -Layer3 | 2048 → 10 | 0.5 | ~16.4M | **TRUE CNN** |
| **Reduced_v2** | -Layer4 | 1024 → 10 | 0.5 | ~10.8M | **TRUE CNN** |
| **Deeper_v1** | +Conv after layer1 | 2048 → 10 | 0.5 | ~26.0M | **TRUE CNN** |
| **Deeper_v2** | +Conv after layer2 | 2048 → 10 | 0.5 | ~26.2M | **TRUE CNN** |
| **Deeper_v3** | +Conv after layer3 | 2048 → 10 | 0.5 | ~26.4M | **TRUE CNN** |

**Backbone Modification Methods:**

1. **Add Convolutional Blocks** (`add_conv_after_layer`):
   - Inserts `[Conv2d → BN → ReLU → Conv2d → BN → ReLU]` after specified layer
   - Increases model depth and capacity
   - Used in Deeper_v1 (layer1), Deeper_v2 (layer2), Deeper_v3 (layer3)

2. **Remove Layers** (`remove_layer`):
   - Sets specified layer to `nn.Identity()`
   - Adjusts subsequent layers to handle channel dimension changes
   - Reduces model depth and parameters
   - Used in Reduced_v1 (remove layer3), Reduced_v2 (remove layer4)

---

### 3. `src/training/` - Training Framework

**File:** `ResNet50_trainer.py`

**Provides:** 
- `TrainingConfig` dataclass (centralized configuration)
- Nine training configurations for all experiments
- `ClassificationTrainer` class

**TrainingConfig Dataclass:**
```python
@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_type: str = 'adamw'
    
    # Training schedule
    epochs: int = 50
    use_scheduler: bool = False
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Loss function
    label_smoothing: float = 0.1
    use_class_weights: bool = False
    
    # Mixed precision
    use_amp: bool = True
    
    # Description
    description: str = 'Default training configuration'
```

**Nine Training Configurations:**

```python
# Baseline
TRAINING_CONFIG_BASELINE = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Baseline training with moderate regularization'
)

# FC_v1 - Enhanced FC with stronger regularization
TRAINING_CONFIG_V1 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=5e-3,              # ↑ Stronger
    epochs=60,                      # ↑ More epochs
    early_stopping_patience=12,     # ↑ Longer patience
    label_smoothing=0.15,           # ↑ Higher smoothing
    description='Enhanced FC head with stronger regularization'
)

# Reduced_v1 - Removed layer3
TRAINING_CONFIG_V3 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Reduced depth backbone (layer3 removed) with standard regularization'
)

# Reduced_v2 - Removed layer4
TRAINING_CONFIG_V4 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Reduced depth backbone (layer4 removed) with standard regularization'
)

# Deeper_v1 - Added conv after layer1
TRAINING_CONFIG_V5 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Deeper backbone (conv added after layer1) with standard regularization'
)

# Deeper_v2 - Added conv after layer2
TRAINING_CONFIG_V6 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Deeper backbone (conv added after layer2) with standard regularization'
)

# Deeper_v3 - Added conv after layer3
TRAINING_CONFIG_V7 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Deeper backbone (conv added after layer3) with standard regularization'
)
```

**Responsibilities:**
- Training loop (forward/backward pass)
- Validation loop
- Optimization (AdamW/Adam/SGD based on config)
- Mixed precision training (AMP)
- Model checkpointing (best model)
- Early stopping (configurable)
- **CSV logging** of training history (every epoch)

**Key Methods:**
- `__init__(model, config)` - Initialize with complete TrainingConfig
- `train_epoch()` - One epoch of training
- `validate()` - Validation on val set
- `train()` - Full training loop with CSV logging

**Usage:**
```python
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    output_dir='outputs/training'
)
# Automatically logs every epoch to training_history.csv
```

**Key Changes from Previous Version:**
- ❌ **Removed:** Multiple hyperparameters in `__init__` and `train()` method
- ✅ **Added:** TrainingConfig dataclass for centralized configuration
- ✅ **Added:** CSV logging for detailed training history analysis
- ✅ **Added:** Flexible optimizer selection (adamw/adam/sgd)
- ✅ **Added:** Configurable scheduler support

---

### 4. `src/data_processing/` - Unified Data Loading

**File:** `ClassificationDataLoader.py` (NEW)

**Provides:**
- `create_classification_dataloaders()` - Main function with configurable augmentation
- `create_baseline_dataloaders()` - Convenience function for standard augmentation
- `create_enhanced_dataloaders()` - Convenience function for enhanced augmentation

**Two Augmentation Strategies:**

```python
# Standard augmentation (Baseline, V3)
'standard':
  - RandomRotation(15°)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - NO RandomAffine

# Enhanced augmentation (V1, V2)
'enhanced':
  - RandomRotation(20°)
  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
  - RandomAffine(translate=(0.1, 0.1))
```

**Responsibilities:**
- Unified data loading across all experiments
- Configurable augmentation strategies
- Consistent validation/test transforms (NO augmentation)
- T4 GPU optimized DataLoader settings

**Usage:**
```python
# For baseline experiments
train_loader, val_loader, test_loader, classes = create_baseline_dataloaders(DATA_ROOT)

# For customized experiments
train_loader, val_loader, test_loader, classes = create_enhanced_dataloaders(DATA_ROOT)
```

**Key Benefits:**
- ✅ Eliminates code duplication (~120 lines saved)
- ✅ Ensures consistency within same augmentation type
- ✅ Easy to modify (single source of truth)
- ✅ Clear intent (function names indicate augmentation type)

---

### 5. `src/evaluation/` - Evaluation Framework

**File:** `classification_evaluator.py`

**Provides:** `ClassificationEvaluator` class

**Responsibilities:**
- Generate predictions on test set
- Calculate metrics (accuracy, precision, recall, F1)
- Create confusion matrix (high-resolution seaborn heatmap)
- **Plot training curves** (loss + accuracy dual-panel)
- **Analyze overfitting/underfitting patterns**
- **Generate comprehensive experiment summary** (abstracted in evaluator)
- Save evaluation results (JSON + report)

**New Methods:**

1. **`plot_training_curves(history, output_dir)`**:
   - Generates dual-panel plot: Loss curves + Accuracy curves
   - Saves as PNG in `visualization/` directory
   - Shows training vs validation performance over epochs

2. **`analyze_overfitting(history)`**:
   - Automatically detects patterns (overfitting/underfitting/good fit)
   - Calculates gap between train and val accuracy
   - Provides recommendations for improvement
   - Returns structured analysis dictionary

3. **`generate_experiment_summary(...)`**:
   - Creates comprehensive markdown summary
   - Includes methodology, configs, results, analysis
   - References training curves and confusion matrix
   - Abstracted from experiment scripts (DRY principle)

**Metrics Calculated:**
- Overall: accuracy, precision, recall, F1 (weighted & macro)
- Per-class: precision, recall, F1
- Confusion matrix (saved as high-res PNG)

**Usage:**
```python
evaluator = ClassificationEvaluator(class_names=class_names)

# Evaluate
metrics = evaluator.evaluate(
    model=model,
    test_loader=test_loader,
    output_dir='outputs/evaluation'
)

# Visualize training
evaluator.plot_training_curves(history['history'], 'outputs/visualization')

# Analyze overfitting
analysis = evaluator.analyze_overfitting(history['history'])

# Generate summary
evaluator.generate_experiment_summary(
    experiment_name='baseline',
    model_config=BASELINE_CONFIG,
    training_config=TRAINING_CONFIG_BASELINE,
    trainer_metrics={'best_val_acc': trainer.best_val_acc},
    evaluation_metrics=metrics,
    overfitting_analysis=analysis,
    output_dir='outputs/'
)
```

**Key Changes from Previous Version:**
- ✅ **Added:** Training curve visualization
- ✅ **Added:** Overfitting/underfitting analysis
- ✅ **Enhanced:** High-resolution confusion matrix
- ✅ **Added:** Comprehensive summary generation
- ✅ **Refactored:** Removed redundant summary code from experiments

---

## Experiment Comparison

### Experiment Design Philosophy

**Critical Methodology Requirements (per teacher):**
1. ✅ **NO layer freezing** - All layers trainable from epoch 1
2. ✅ **True CNN customization** - Must modify backbone structure, not just hyperparameters
3. ✅ **Consistent dataset splits** - Same split used across all experiments
4. ✅ **Single-phase training** - Simplified training pipeline

### Nine Experiments Overview

| Experiment | Model Config | Training Config | Customization Type | Key Features |
|------------|-------------|-----------------|-------------------|--------------|
| **Baseline** | BASELINE_CONFIG | TRAINING_CONFIG_BASELINE | None (control) | Standard ResNet50 |
| **GridSearch** | BASELINE_CONFIG | Variable configs | Hyperparameter optimization | Systematic LR/WD/LS search |
| **FC_v1** | CUSTOMIZED_V1_CONFIG | TRAINING_CONFIG_V1 | FC enhancement | Multi-layer FC + BN + dropout 0.7 |
| **Reduced_v1** | CUSTOMIZED_V3_CONFIG | TRAINING_CONFIG_V3 | **TRUE CNN** | Removed layer3 (~16.4M params) |
| **Reduced_v2** | CUSTOMIZED_V4_CONFIG | TRAINING_CONFIG_V4 | **TRUE CNN** | Removed layer4 (~10.8M params) |
| **Deeper_v1** | CUSTOMIZED_V5_CONFIG | TRAINING_CONFIG_V5 | **TRUE CNN** | +Conv after layer1 (~26.0M params) |
| **Deeper_v2** | CUSTOMIZED_V6_CONFIG | TRAINING_CONFIG_V6 | **TRUE CNN** | +Conv after layer2 (~26.2M params) |
| **Deeper_v3** | CUSTOMIZED_V7_CONFIG | TRAINING_CONFIG_V7 | **TRUE CNN** | +Conv after layer3 (~26.4M params) |

### Detailed Comparison

#### **Baseline Experiment**
- **Model:** Standard ResNet50 with single FC layer (2048→10)
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Control group for comparison

#### **GridSearch Experiment**
- **Model:** Standard ResNet50 (same as baseline)
- **Training:** Tests 27 combinations of LR × WD × Label Smoothing
- **Purpose:** Find optimal hyperparameters for baseline configuration
- **Output:** CSV file with all results and best configuration identified

#### **FC_v1 Experiment (FC Enhancement)**
- **Model:** Enhanced FC head (2048→512→256→10) with BatchNorm
- **Training:** 60 epochs, lr=1e-4, weight_decay=5e-3, patience=12
- **Purpose:** Test if deeper FC head improves performance
- **Note:** NOT true CNN customization (only FC layers modified)

#### **Reduced_v1 Experiment (TRUE CNN Customization)**
- **Model:** Removed layer3 (reduced depth) with standard FC head
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Test if reduced depth affects performance (lighter model ~16.4M)
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

#### **Reduced_v2 Experiment (TRUE CNN Customization)**
- **Model:** Removed layer4 (further reduced depth) with adjusted FC (1024→10)
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Test even lighter model (~10.8M params)
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

#### **Deeper_v1 Experiment (TRUE CNN Customization)**
- **Model:** Added convolutional blocks after layer1 with standard FC head
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Test if adding conv at early layer improves shallow feature extraction
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

#### **Deeper_v2 Experiment (TRUE CNN Customization)**
- **Model:** Added convolutional blocks after layer2 with standard FC head
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Test if adding conv at mid layer improves feature extraction
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

#### **Deeper_v3 Experiment (TRUE CNN Customization)**
- **Model:** Added convolutional blocks after layer3 with standard FC head
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Purpose:** Test if adding conv at late layer improves high-level feature extraction
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

### Fair Comparison Strategy

**Controlled Comparisons:**

1. **Reduced Models (v1 vs v2):**
   - Both reduce depth but at different layers
   - Compare impact of removing layer3 vs layer4
   - Assess trade-off between model size and accuracy

2. **Deeper Models (v1 vs v2 vs v3):**
   - All add conv blocks but at different positions
   - Controlled comparison: which layer benefits most from additional conv?
   - Isolates backbone modification impact (all use single FC head like baseline)

3. **FC Enhancement (FC_v1):**
   - Tests if deeper classifier helps without backbone changes
   - NOT true CNN customization but useful reference point

4. **Baseline vs All Variants:**
   - Baseline serves as control group
   - All improvements measured against this reference

---

## File Structure

```
CNN_A2/
├── experiments/                          [Flow Control]
│   ├── classification_ResNet50_baseline.py           (Baseline)
│   ├── classification_ResNet50_baseline_gridsearch.py (GridSearch)
│   ├── classification_ResNet50_FC_v1.py              (FC Enhancement v1)
│   ├── classification_ResNet50_reduced_v1.py         (Reduced Depth v1: -layer3)
│   ├── classification_ResNet50_reduced_v2.py         (Reduced Depth v2: -layer4)
│   ├── classification_ResNet50_deeper_v1.py          (Deeper Backbone v1: +conv layer1)
│   ├── classification_ResNet50_deeper_v2.py          (Deeper Backbone v2: +conv layer2)
│   └── classification_ResNet50_deeper_v3.py          (Deeper Backbone v3: +conv layer3)
│
├── src/
│   ├── models/                         [Model Definitions]
│   │   ├── __init__.py
│   │   └── ResNet50ClassifierModel.py      (ResNet50 + 9 configs + backbone mod)
│   │
│   ├── training/                       [Training Framework]
│   │   ├── __init__.py
│   │   └── ResNet50_trainer.py       (TrainingConfig + Trainer + CSV logging)
│   │
│   ├── data_processing/                [Unified Data Loading]
│   │   ├── __init__.py
│   │   ├── classification_split.py         (Dataset splitting)
│   │   └── ClassificationDataLoader.py     (Unified loader with 2 aug strategies)
│   │
│   └── evaluation/                     [Evaluation Framework]
│       ├── __init__.py
│       └── classification_evaluator.py     (Metrics + visualization + analysis + summary)
│
├── data/25509225/Image_Classification/split_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
└── outputs/                            [Results]
    ├── classification_baseline_TIMESTAMP/
    │   ├── training/
    │   │   ├── best_model.pth
    │   │   └── training_history.csv          ← Detailed epoch-by-epoch metrics
    │   ├── evaluation/
    │   │   ├── evaluation_metrics.json
    │   │   ├── classification_report.txt
    │   │   └── confusion_matrix.png          ← High-res heatmap
    │   ├── visualization/                     
    │   │   └── training_curves.png           ← Loss + accuracy plots
    │   └── experiment_summary.md             ← Comprehensive summary
    │
    ├── classification_FC_v1_TIMESTAMP/
    │   └── ... (same structure)
    │
    ├── classification_reduced_v1_TIMESTAMP/
    │   └── ... (same structure)
    │
    ├── classification_reduced_v2_TIMESTAMP/
    │   └── ... (same structure)
    │
    ├── classification_deeper_v1_TIMESTAMP/
    │   └── ... (same structure)
    │
    ├── classification_deeper_v2_TIMESTAMP/
    │   └── ... (same structure)
    │
    └── classification_deeper_v3_TIMESTAMP/
        └── ... (same structure)
```

---

## Running Experiments

### Quick Start

```bash
# Run baseline experiment
python experiments/classification_ResNet50_baseline.py

# Run hyperparameter grid search (optional, time-consuming)
python experiments/classification_ResNet50_baseline_gridsearch.py

# Run FC enhancement experiment
python experiments/classification_ResNet50_FC_v1.py

# Run reduced depth experiments
python experiments/classification_ResNet50_reduced_v1.py
python experiments/classification_ResNet50_reduced_v2.py

# Run deeper backbone experiments
python experiments/classification_ResNet50_deeper_v1.py
python experiments/classification_ResNet50_deeper_v2.py
python experiments/classification_ResNet50_deeper_v3.py
```

### Command Line Options

All experiments support the following options:

```bash
# Use pretrained weights (default: True from config)
python experiments/classification_ResNet50_[experiment].py --pretrained False

# Specify data augmentation strategy (default: 'none')
python experiments/classification_ResNet50_[experiment].py --dataAugmentation enhanced

# Combine options
python experiments/classification_ResNet50_baseline.py --pretrained True --dataAugmentation standard
```

**Options:**
- `--pretrained`: `True` (use ImageNet weights), `False` (train from scratch), or omit (use config default)
- `--dataAugmentation`: `none` (basic preprocessing), `standard` (rotation 15° + color jitter), `enhanced` (rotation 20° + stronger jitter + affine)

### Expected Output

Each experiment creates a timestamped directory in `outputs/` containing:
- **Trained model** (`training/best_model.pth`)
- **Training history CSV** (`training/training_history.csv`) - Epoch-by-epoch metrics
- **Evaluation metrics** (`evaluation/evaluation_metrics.json`)
- **Classification report** (`evaluation/classification_report.txt`)
- **Confusion matrix** (`evaluation/confusion_matrix.png`) - High-resolution heatmap
- **Training curves** (`visualization/training_curves.png`) - Loss + accuracy plots
- **Experiment summary** (`experiment_summary.md`) - Comprehensive markdown report

### Post-Training Analysis

You can analyze the CSV files for custom insights:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
df = pd.read_csv('outputs/classification_baseline/run_XXX/training/training_history.csv')

# Plot custom visualizations
plt.figure(figsize=(12, 5))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training History')
plt.show()

# Compare experiments
experiments = ['baseline', 'FC_v1', 'reduced_v1', 'reduced_v2', 
               'deeper_v1', 'deeper_v2', 'deeper_v3']
for exp in experiments:
    # Find latest run and load CSV
    # Plot comparison curves
```

---

## Key Design Principles

1. **Separation of Concerns:**
   - Experiments control the flow
   - Models define architecture
   - Training handles optimization (with centralized config)
   - Data loading is unified and consistent
   - Evaluation calculates metrics and generates reports

2. **Centralized Configuration:**
   - All model configs in one place (9 configurations)
   - All training configs in one place (9 TrainingConfig objects)
   - Easy to compare and modify

3. **No Code Duplication (DRY Principle):**
   - Unified data loader eliminates ~120 lines of duplicate code
   - Abstracted summary generation removes redundancy from experiments
   - TrainingConfig eliminates scattered hyperparameter definitions

4. **Teacher Compliance:**
   - ✅ NO layer freezing (all layers trainable)
   - ✅ True CNN customization (backbone modification in reduced/deeper variants)
   - ✅ Consistent dataset splits (unified loader)
   - ✅ Single-phase training (simplified pipeline)
   - ✅ Methodology focus documented in summaries

5. **Reproducibility:**
   - Fixed random seeds (student ID)
   - Deterministic data splitting
   - Saved configurations (TrainingConfig has description field)
   - CSV logging for detailed training history

6. **Comparability:**
   - Same data loaders within augmentation type
   - Consistent evaluation metrics
   - Side-by-side result comparison enabled
   - Training curves for visual comparison
   - Controlled experiments: Deeper_v1/v2/v3 isolate layer position impact

---

## Major Refactoring Summary

### What Changed Since Initial Version

1. **Removed Layer Freezing:**
   - Deleted `freeze_backbone()` and `unfreeze_backbone()` methods
   - Converted from two-phase to single-phase training
   - All layers trainable from epoch 1

2. **Expanded True CNN Customization:**
   - Added backbone modification capability to ResNet50Classifier
   - **Reduced Models:** Remove layer3 (reduced_v1) or layer4 (reduced_v2)
   - **Deeper Models:** Add conv blocks after layer1/2/3 (deeper_v1/v2/v3)
   - Renamed experiments for clarity: v1→FC_v1, v3→reduced_v1, etc.
   - Deleted old v2 experiment (combined backbone+FC enhancement)

3. **Centralized Training Configuration:**
   - Created TrainingConfig dataclass
   - Defined 9 independent training configurations
   - Simplified Trainer API (removed scattered parameters)

4. **Unified Data Loading:**
   - Created ClassificationDataLoader module
   - Two augmentation strategies (standard/enhanced)
   - Eliminated code duplication across experiments

5. **Enhanced Evaluation:**
   - Added training curve visualization
   - Added overfitting/underfitting analysis
   - Enhanced confusion matrix (high-res)
   - Abstracted summary generation to evaluator

6. **Added CSV Logging:**
   - Every epoch's metrics logged to CSV
   - Enables detailed post-training analysis
   - Easy to import into Pandas/Excel

7. **Improved Experiment Naming:**
   - Clear categorization: Baseline, FC Enhancement, Reduced Depth, Deeper Backbone
   - Systematic naming enables easy comparison within categories
   - Better reflects experimental purpose and methodology

---

## Next Steps

1. ✅ Run all experiments (baseline, gridsearch, FC_v1, reduced_v1/v2, deeper_v1/v2/v3)
2. ✅ Compare results in `outputs/` directories
3. ✅ Analyze `experiment_summary.md` files for each variant
4. ✅ Review training curves for overfitting patterns
5. ✅ Examine confusion matrices for class-specific performance
6. ✅ Generate comprehensive comparison across:
   - Reduced depth models (which layer removal is less harmful?)
   - Deeper backbone models (which layer addition helps most?)
   - FC enhancement vs baseline (does deeper classifier help?)
7. ✅ Include comprehensive comparison in assignment report
8. ✅ Discuss methodology compliance and design decisions

### Key Analysis Questions:

**Reduced Models:**
- Does removing layer3 or layer4 cause more accuracy drop?
- Is the parameter reduction worth the performance trade-off?
- Which classes suffer most from reduced depth?

**Deeper Models:**
- Which layer position benefits most from additional conv blocks?
- Does adding depth at early/mid/late stages affect different features?
- Is there diminishing returns as we add more layers?

**FC Enhancement:**
- Does multi-layer FC head improve performance without backbone changes?
- Is the increased complexity justified by accuracy gains?

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Last Major Update:** 2026-05-04 - Renamed experiments for clarity, expanded to 9 variants with systematic categorization (Baseline, FC Enhancement, Reduced Depth, Deeper Backbone)
