# Classification Task Architecture - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-05-01  

---

## Architecture Overview

The classification task follows a clean, modular architecture with clear separation of concerns and centralized configuration management:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS (Flow Control)               │
│                                                               │
│  classification_ResNet50_baseline.py    (Baseline)           │
│  classification_ResNet50_v1.py          (Customized V1)      │
│  classification_ResNet50_v2.py          (Customized V2)      │
│  classification_ResNet50_v3.py          (Customized V3)      │
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
- `classification_ResNet50_baseline.py` - Baseline experiment (standard ResNet50)
- `classification_ResNet50_v1.py` - Customized V1 (enhanced FC head)
- `classification_ResNet50_v2.py` - Customized V2 (CNN backbone modification)
- `classification_ResNet50_v3.py` - Customized V3 (reduced depth)

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
from src.training.classification_trainer import ClassificationTrainer, TRAINING_CONFIG_BASELINE
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
- Four configuration dictionaries for different experiments

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

# Customized V1 (classification_ResNet50_v1.py)
CUSTOMIZED_V1_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.7,                # Higher dropout
    'pretrained': True,
    'additional_fc_layers': True,       # Multi-layer FC with BatchNorm
    'use_batch_norm': True,
    'modify_backbone': False            # Standard backbone
}

# Customized V2 (classification_ResNet50_v2.py) - TRUE CNN CUSTOMIZATION
CUSTOMIZED_V2_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.6,
    'pretrained': True,
    'additional_fc_layers': True,       # Multi-layer FC with BatchNorm
    'use_batch_norm': True,
    'modify_backbone': True,            # ✓ Modify backbone!
    'remove_layer': None,
    'add_conv_after_layer': 'layer2'    # Add conv blocks after layer2
}

# Customized V3 (classification_ResNet50_v3.py) - TRUE CNN CUSTOMIZATION
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
```

**Architecture Differences:**

| Component | Baseline | V1 (FC Enhanced) | V2 (Backbone+) | V3 (Reduced) |
|-----------|----------|------------------|----------------|--------------|
| Backbone | Standard | Standard | +Conv after layer2 | -Layer3 |
| Classifier Head | 2048 → 10 | 2048→512→256→10 + BN | 2048→512→256→10 + BN | 2048 → 10 |
| BatchNorm in FC | No | Yes | Yes | No |
| Dropout | 0.5 | 0.7 | 0.6 | 0.5 |
| Total Params | ~25.6M | ~25.6M | ~26.2M | ~16.4M |
| Customization Type | None | FC only | **TRUE CNN** | **TRUE CNN** |

**Backbone Modification Methods:**

1. **Add Convolutional Blocks** (`add_conv_after_layer`):
   - Inserts `[Conv2d → BN → ReLU → Conv2d → BN → ReLU]` after specified layer
   - Increases model depth and capacity
   - Used in V2 (after layer2)

2. **Remove Layers** (`remove_layer`):
   - Sets specified layer to `nn.Identity()`
   - Adjusts subsequent layers to handle channel dimension changes
   - Reduces model depth and parameters
   - Used in V3 (remove layer3)

---

### 3. `src/training/` - Training Framework

**File:** `classification_trainer.py`

**Provides:** 
- `TrainingConfig` dataclass (centralized configuration)
- Four training configurations
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

**Four Training Configurations:**

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

# V1 - Enhanced FC with stronger regularization
TRAINING_CONFIG_V1 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=5e-3,              # ↑ Stronger
    epochs=60,                      # ↑ More epochs
    early_stopping_patience=12,     # ↑ Longer patience
    label_smoothing=0.15,           # ↑ Higher smoothing
    description='Enhanced FC head with stronger regularization'
)

# V2 - CNN backbone modification with enhanced FC
TRAINING_CONFIG_V2 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=5e-3,
    epochs=60,
    early_stopping_patience=12,
    label_smoothing=0.15,
    description='CNN backbone modification with enhanced FC head and strong regularization'
)

# V3 - Reduced depth backbone
TRAINING_CONFIG_V3 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=50,
    early_stopping_patience=10,
    label_smoothing=0.1,
    description='Reduced depth backbone with standard regularization'
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
- **Generate comprehensive experiment summary** (markdown)
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

### Four Experiments Overview

| Experiment | Model Config | Training Config | Augmentation | Customization Type | Key Features |
|------------|-------------|-----------------|--------------|-------------------|--------------|
| **Baseline** | BASELINE_CONFIG | TRAINING_CONFIG_BASELINE | Standard | None (control) | Standard ResNet50 |
| **V1** | CUSTOMIZED_V1_CONFIG | TRAINING_CONFIG_V1 | Enhanced | FC enhancement | Multi-layer FC + BN |
| **V2** | CUSTOMIZED_V2_CONFIG | TRAINING_CONFIG_V2 | Enhanced | **TRUE CNN** | +Conv after layer2 |
| **V3** | CUSTOMIZED_V3_CONFIG | TRAINING_CONFIG_V3 | Standard | **TRUE CNN** | -Layer3 (reduced) |

### Detailed Comparison

#### **Baseline Experiment**
- **Model:** Standard ResNet50 with single FC layer (2048→10)
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Augmentation:** Standard (rotation 15°, color jitter 0.2)
- **Purpose:** Control group for comparison

#### **V1 Experiment (FC Enhancement)**
- **Model:** Enhanced FC head (2048→512→256→10) with BatchNorm
- **Training:** 60 epochs, lr=1e-4, weight_decay=5e-3, patience=12
- **Augmentation:** Enhanced (rotation 20°, color jitter 0.3+hue, random affine)
- **Purpose:** Test if deeper FC head improves performance
- **Note:** NOT true CNN customization (only FC layers modified)

#### **V2 Experiment (TRUE CNN Customization)**
- **Model:** Added convolutional blocks after layer2 + enhanced FC head
- **Training:** 60 epochs, lr=1e-4, weight_decay=5e-3, patience=12
- **Augmentation:** Enhanced (same as V1)
- **Purpose:** Test if backbone modification improves feature extraction
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

#### **V3 Experiment (TRUE CNN Customization)**
- **Model:** Removed layer3 (reduced depth) with standard FC head
- **Training:** 50 epochs, lr=1e-4, weight_decay=1e-4, patience=10
- **Augmentation:** Standard (same as baseline)
- **Purpose:** Test if reduced depth affects performance (lighter model)
- **Compliance:** ✅ Meets teacher's requirement for true CNN customization

### Fair Comparison Strategy

**Within Same Augmentation Type:**
- **Baseline vs V3:** Both use standard augmentation → Direct comparison valid
  - Difference is purely due to model architecture (standard vs reduced depth)
- **V1 vs V2:** Both use enhanced augmentation → Direct comparison valid
  - Difference is purely due to model architecture (FC only vs backbone+FC)

**Cross Augmentation Types:**
- Comparisons between standard/enhanced groups should note that differences may come from both model AND augmentation
- This is intentional: tests if complex models benefit more from stronger augmentation

---

## File Structure

```
CNN_A2/
├── experiments/                          [Flow Control]
│   ├── classification_ResNet50_baseline.py   (Baseline)
│   ├── classification_ResNet50_v1.py         (Customized V1)
│   ├── classification_ResNet50_v2.py         (Customized V2)
│   └── classification_ResNet50_v3.py         (Customized V3)
│
├── src/
│   ├── models/                         [Model Definitions]
│   │   ├── __init__.py
│   │   └── ResNet50ClassifierModel.py      (ResNet50 + 4 configs + backbone mod)
│   │
│   ├── training/                       [Training Framework]
│   │   ├── __init__.py
│   │   └── classification_trainer.py       (TrainingConfig + Trainer + CSV logging)
│   │
│   ├── data_processing/                [Unified Data Loading] ← NEW
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
    │   │   └── training_history.csv          ← NEW: Detailed epoch-by-epoch metrics
    │   ├── evaluation/
    │   │   ├── evaluation_metrics.json
    │   │   ├── classification_report.txt
    │   │   └── confusion_matrix.png          ← ENHANCED: High-res heatmap
    │   ├── visualization/                     ← NEW DIRECTORY
    │   │   └── training_curves.png           ← NEW: Loss + accuracy plots
    │   └── experiment_summary.md             ← ENHANCED: Comprehensive summary
    │
    ├── classification_customized_v1_TIMESTAMP/
    │   └── ... (same structure)
    │
    ├── classification_customized_v2_TIMESTAMP/
    │   └── ... (same structure)
    │
    └── classification_customized_v3_TIMESTAMP/
        └── ... (same structure)
```

---

## Running Experiments

### Quick Start

```bash
# Run all 4 experiments
python experiments/classification_ResNet50_baseline.py
python experiments/classification_ResNet50_v1.py
python experiments/classification_ResNet50_v2.py
python experiments/classification_ResNet50_v3.py
```

### Expected Output

Each experiment creates a timestamped directory in `outputs/` containing:
- **Trained model** (`training/best_model.pth`)
- **Training history CSV** (`training/training_history.csv`) - NEW!
- **Evaluation metrics** (`evaluation/evaluation_metrics.json`)
- **Classification report** (`evaluation/classification_report.txt`)
- **Confusion matrix** (`evaluation/confusion_matrix.png`) - ENHANCED!
- **Training curves** (`visualization/training_curves.png`) - NEW!
- **Experiment summary** (`experiment_summary.md`) - ENHANCED!

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
experiments = ['baseline', 'customized_v1', 'customized_v2', 'customized_v3']
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
   - All model configs in one place (4 configurations)
   - All training configs in one place (4 TrainingConfig objects)
   - Easy to compare and modify

3. **No Code Duplication (DRY Principle):**
   - Unified data loader eliminates ~120 lines of duplicate code
   - Abstracted summary generation removes redundancy from experiments
   - TrainingConfig eliminates scattered hyperparameter definitions

4. **Teacher Compliance:**
   - ✅ NO layer freezing (all layers trainable)
   - ✅ True CNN customization (backbone modification in V2/V3)
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

---

## Major Refactoring Summary

### What Changed Since Initial Version

1. **Removed Layer Freezing:**
   - Deleted `freeze_backbone()` and `unfreeze_backbone()` methods
   - Converted from two-phase to single-phase training
   - All layers trainable from epoch 1

2. **Added True CNN Customization:**
   - Added backbone modification capability to ResNet50Classifier
   - V2: Adds convolutional blocks after layer2
   - V3: Removes layer3 with proper channel adjustment
   - Deleted old CUSTOMIZED_CONFIG (only modified FC, not compliant)

3. **Centralized Training Configuration:**
   - Created TrainingConfig dataclass
   - Defined 4 independent training configurations
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

---

## Next Steps

1. ✅ Run all 4 experiments
2. ✅ Compare results in `outputs/` directories
3. ✅ Analyze `experiment_summary.md` files
4. ✅ Review training curves for each experiment
5. ✅ Examine overfitting analysis findings
6. ✅ Include comprehensive comparison in assignment report
7. ✅ Discuss methodology compliance and design decisions

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Last Major Update:** 2026-05-01 - Complete architecture refactoring with centralized configuration and unified data loading
