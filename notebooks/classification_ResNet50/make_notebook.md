# ResNet50 Classification Experiments - Notebook Creation Plan

**Created:** 2026-05-06  
**Student ID:** 25509225  
**Project:** CNN Assignment 2

---

## 📋 Overview

Convert existing Python experiment scripts into self-contained Jupyter Notebooks for submission. All outputs must be displayed inline in the notebook, with minimal file saving (only best_model.pth).

---

## 🎯 Core Requirements

### **Display Requirements**
✅ All visualizations must be displayed inline in notebook output cells  
✅ All text/metrics must be printed to notebook output  
❌ NO external CSV files (training_history.csv)  
❌ NO external PNG files (confusion_matrix.png, training_curves.png)  
❌ NO JSON/TXT report files  
❌ NO experiment_summary.md  

### **File Saving Requirements**
✅ Save ONLY: `outputs/classification_[experiment]/best_model.pth`  
❌ Do NOT save: final_model.pth  
❌ Do NOT save: any intermediate files  

### **Default Configuration**
- **Data Augmentation:** `'none'` (can be modified by editing cell variables)
- **Pretrained:** `True` (can be modified by editing cell variables)
- **Batch Size:** `16`
- **Output Directory:** `outputs/classification_[experiment]/` (simplified, no timestamp subdirectory)

---

## 🗂️ File Structure

```
notebooks/classification_ResNet50/
├── make_notebook.md                    # This file (execution plan)
├── ResNet50_modules.ipynb             # Shared modules library
├── classification_ResNet50_baseline.ipynb      # Baseline experiment
├── classification_ResNet50_reduced_v1.ipynb    # Reduced v1 experiment
└── classification_ResNet50_deeper_v1.ipynb     # Deeper v1 experiment
```

---

## 📝 Execution Steps

### **Step 1: Read Source Code Files**

Read and understand the following source files:

1. `src/models/ResNet50ClassifierModel.py`
   - ResNet50Classifier class
   - BASELINE_CONFIG
   - CUSTOMIZED_REDUCED_V1_CONFIG
   - CUSTOMIZED_DEEPER_V1_CONFIG
   - Backbone modification logic

2. `src/training/ResNet50_trainer.py`
   - TrainingConfig dataclass
   - TRAINING_CONFIG_BASELINE
   - TRAINING_CONFIG_REDUCED_V1
   - TRAINING_CONFIG_DEEPER_V1
   - ClassificationTrainer class

3. `src/data_processing/ClassificationDataLoader.py`
   - create_classification_dataloaders()
   - Transform definitions
   - DataLoader creation logic

4. `src/evaluation/classification_evaluator.py`
   - ClassificationEvaluator class
   - evaluate() method
   - calculate_metrics()
   - plot_confusion_matrix()
   - analyze_overfitting()

5. `src/utils/file_utils.py` (if needed)
   - Utility functions

**After completion:** Confirm with user before proceeding to Step 2.

---

### **Step 2: Create ResNet50_modules.ipynb**

Create a comprehensive modules notebook containing all shared code.

#### **Cell Structure:**

**Cell 1: Imports & Setup**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os

# Jupyter magic commands
%matplotlib inline

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
```

**Cell 2: Model Definition**
```python
# Complete ResNet50Classifier class definition
# Include all backbone modification logic
# Include configuration dictionaries:
# - BASELINE_CONFIG
# - CUSTOMIZED_REDUCED_V1_CONFIG
# - CUSTOMIZED_DEEPER_V1_CONFIG
```

**Cell 3: Training Framework**
```python
# TrainingConfig dataclass
# Training configuration dictionaries:
# - TRAINING_CONFIG_BASELINE
# - TRAINING_CONFIG_REDUCED_V1
# - TRAINING_CONFIG_DEEPER_V1

# ClassificationTrainer class
# IMPORTANT MODIFICATIONS:
# - train() method should NOT save CSV files
# - train() method should ONLY save best_model.pth
# - Return history dict without file I/O
```

**Cell 4: Data Loading**
```python
# Transform definitions
# create_classification_dataloaders() function
# Helper functions:
# - create_baseline_dataloaders()
# - create_enhanced_dataloaders()
```

**Cell 5: Evaluation Framework**
```python
# ClassificationEvaluator class
# IMPORTANT MODIFICATIONS:
# - evaluate() returns metrics dict WITHOUT saving files
# - plot_confusion_matrix() returns fig object for inline display
# - analyze_overfitting() returns analysis dict
# - REMOVE generate_experiment_summary() method
# - REMOVE plot_training_curves() method (will be done in experiment notebooks)
```

**Cell 6: Utility Functions**
```python
# Helper functions for formatting output
# Print utilities
```

**Key Modifications from Source Code:**

1. **ClassificationTrainer.train():**
   ```python
   # Remove: df.to_csv(os.path.join(output_dir, 'training_history.csv'))
   # Keep: torch.save({...}, os.path.join(output_dir, 'best_model.pth'))
   # Return: {'history': history_list}
   ```

2. **ClassificationEvaluator.evaluate():**
   ```python
   # Remove: json.dump(), txt file writing, PNG saving
   # Return: metrics dict (including confusion_matrix as numpy array)
   ```

3. **Removed Methods:**
   - `plot_training_curves()` → Will be implemented directly in experiment notebooks
   - `generate_experiment_summary()` → Replaced with print statements in experiments

**After completion:** Confirm with user before proceeding to Step 3.

---

### **Step 3: Create classification_ResNet50_baseline.ipynb**

Convert `experiments/classification_ResNet50_baseline.py` to notebook format.

#### **Cell Structure:**

**Cell 1: Load Modules**
```python
%run ./ResNet50_modules.ipynb
```

**Cell 2: Configuration**
```python
STUDENT_ID = "25509225"
DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
BATCH_SIZE = 16
AUGMENTATION_TYPE = 'none'  # Can be changed to 'standard' or 'enhanced'
USE_PRETRAINED = True       # Can be changed to False

output_dir = Path(f'outputs/classification_baseline')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPERIMENT: ResNet50 Baseline")
print("=" * 80)
print(f'\nConfiguration:')
print(f'  Data Root: {DATA_ROOT}')
print(f'  Batch Size: {BATCH_SIZE}')
print(f'  Augmentation: {AUGMENTATION_TYPE}')
print(f'  Pretrained: {USE_PRETRAINED}')
print(f'  Output Dir: {output_dir}')
```

**Cell 3: Step 1 - Load Data**
```python
print("\n[1/5] Loading data...")
train_loader, val_loader, test_loader, class_names = create_classification_dataloaders(
    DATA_ROOT, 
    batch_size=BATCH_SIZE, 
    augmentation_type=AUGMENTATION_TYPE
)
print(f'Classes: {class_names}')
print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

aug_descriptions = {
    'none': 'No augmentation (basic preprocessing only)',
    'standard': 'Standard (Rotation 15°, ColorJitter 0.2)',
    'enhanced': 'Enhanced (Rotation 20°, ColorJitter 0.3+hue, RandomAffine)'
}
print(f'Data augmentation: {aug_descriptions[AUGMENTATION_TYPE]}')
```

**Cell 4: Step 2 - Initialize Model**
```python
print("\n[2/5] Initializing model...")

model_config = BASELINE_CONFIG.copy()
model_config['pretrained'] = USE_PRETRAINED

if model_config['pretrained']:
    print('Architecture: Standard ResNet50 with ALL layers trainable (NO freezing)')
    print('Pretrained: YES (ImageNet weights)')
else:
    print('Architecture: Standard ResNet50 with ALL layers trainable (NO freezing)')
    print('Pretrained: NO (Training from scratch)')

model = ResNet50Classifier(**model_config)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')

# Print model summary
trainer_temp = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)
trainer_temp.print_model_summary()
```

**Cell 5: Step 3 - Train**
```python
print("\n[3/5] Training...")
trainer = trainer_temp  # Reuse trainer from Cell 4
criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_BASELINE.label_smoothing)

history = trainer.train(
    train_loader, 
    val_loader, 
    criterion,
    str(output_dir)  # Only saves best_model.pth
)
print(f'Best Val Acc: {trainer.best_val_acc:.4f}')
```

**Cell 6: Load Best Model**
```python
print("\nLoading best model for evaluation...")
best_model_path = output_dir / 'best_model.pth'
if best_model_path.exists():
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(trainer.device)
    print(f'✓ Loaded best model from epoch {checkpoint["epoch"]} (Val Acc: {checkpoint["val_acc"]:.4f})')
else:
    print('Warning: Best model checkpoint not found, using current model')
```

**Cell 7: Step 4 - Evaluate**
```python
print("\n[4/5] Evaluating on test set...")
evaluator = ClassificationEvaluator(class_names)
metrics = evaluator.evaluate(model, test_loader)  # No output_dir parameter

print("\n=== Test Set Metrics ===")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
print(f"\nPrecision (macro): {metrics['precision_macro']:.4f}")
print(f"Recall (macro): {metrics['recall_macro']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
```

**Cell 8: Display Confusion Matrix**
```python
print("\n=== Confusion Matrix ===")
fig, ax = plt.subplots(figsize=(10, 8))
cm_array = metrics['confusion_matrix']
sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

**Cell 9: Step 5 - Display Training Curves**
```python
print("\n[5/5] Training History Analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history['history']['epoch'], history['history']['train_loss'], 
             label='Train Loss', linewidth=2)
axes[0].plot(history['history']['epoch'], history['history']['val_loss'], 
             label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(history['history']['epoch'], history['history']['train_acc'], 
             label='Train Acc', linewidth=2)
axes[1].plot(history['history']['epoch'], history['history']['val_acc'], 
             label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training & Validation Accuracy', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Cell 10: Overfitting Analysis**
```python
print("\n=== Overfitting Analysis ===")
analysis = evaluator.analyze_overfitting(history['history'])
print(f"Pattern: {analysis['pattern']}")
print(f"Train-Val Accuracy Gap: {analysis['gap']:.4f}")
print(f"Recommendation: {analysis['recommendation']}")
```

**Cell 11: Final Summary**
```python
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED")
print("=" * 80)
print(f"\nExperiment: ResNet50 Baseline")
print(f"Architecture: Standard ResNet50 with single FC layer (2048→10)")
print(f"Pretrained: {model_config['pretrained']}")
print(f"Data Augmentation: {AUGMENTATION_TYPE}")
print(f"Best Val Accuracy: {trainer.best_val_acc:.4f}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"\nResults saved to: {output_dir}")
print(f"  - best_model.pth ✓")
```

**After completion:** Confirm with user before proceeding to Step 4.

---

### **Step 4: Create classification_ResNet50_reduced_v1.ipynb**

Same structure as baseline, with these changes:

**Cell 2: Configuration**
```python
experiment_name = 'reduced_v1'
output_dir = Path(f'outputs/classification_{experiment_name}')
```

**Cell 4: Initialize Model**
```python
model_config = CUSTOMIZED_REDUCED_V1_CONFIG.copy()
model_config['pretrained'] = USE_PRETRAINED

print('TRUE CNN Customizations:')
print('  1. Backbone modification: Removed layer3 (reduced depth)')
print('  2. Standard FC head: 2048 → 10')
print('  3. Moderate dropout: 0.5')
print('  4. ALL layers trainable (NO freezing)')

if model_config['pretrained']:
    print('Pretrained: YES (ImageNet weights)')
else:
    print('Pretrained: NO (Training from scratch)')
```

**Cell 5: Train**
```python
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_REDUCED_V1)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_REDUCED_V1.label_smoothing)
```

**Cell 11: Final Summary**
```python
print(f"\nExperiment: ResNet50 Reduced v1")
print(f"Architecture: ResNet50 with layer3 removed (~16.4M params)")
print(f"Customization: TRUE CNN - Backbone modification (removed layer3)")
```

**After completion:** Confirm with user before proceeding to Step 5.

---

### **Step 5: Create classification_ResNet50_deeper_v1.ipynb**

Same structure as baseline, with these changes:

**Cell 2: Configuration**
```python
experiment_name = 'deeper_v1'
output_dir = Path(f'outputs/classification_{experiment_name}')
```

**Cell 4: Initialize Model**
```python
model_config = CUSTOMIZED_DEEPER_V1_CONFIG.copy()
model_config['pretrained'] = USE_PRETRAINED

print('TRUE CNN Customizations:')
print('  1. Backbone modification: Added conv blocks after layer1')
print('  2. Single FC head: 2048 → 10 (like Baseline)')
print('  3. Standard dropout: 0.5')
print('  4. ALL layers trainable (NO freezing)')
print('\nPurpose: Test if adding conv blocks at early layer improves performance')

if model_config['pretrained']:
    print('Pretrained: YES (ImageNet weights)')
else:
    print('Pretrained: NO (Training from scratch)')
```

**Cell 5: Train**
```python
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_DEEPER_V1)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_DEEPER_V1.label_smoothing)
```

**Cell 11: Final Summary**
```python
print(f"\nExperiment: ResNet50 Deeper v1")
print(f"Architecture: ResNet50 with conv blocks added after layer1 (~26.0M params)")
print(f"Customization: TRUE CNN - Backbone modification (added conv after layer1)")
```

**After completion:** Confirm with user that all notebooks are created.

---

## 🔧 Key Code Modifications Required

### **1. ClassificationTrainer.train() Method**

**Original behavior:**
- Saves CSV file: `training_history.csv`
- Saves model checkpoint: `best_model.pth`

**Modified behavior:**
- ❌ Remove CSV saving
- ✅ Keep model checkpoint saving
- Return history dict

**Implementation:**
```python
def train(self, train_loader, val_loader, criterion, output_dir=None):
    """
    Training loop with CSV logging.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        output_dir: Directory to save model checkpoints (CSV saving removed)
    
    Returns:
        dict: Training history containing epoch-by-epoch metrics
    """
    # ... existing training loop code ...
    
    # At the end of each epoch, append to history_list
    history_list.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rate': current_lr
    })
    
    # REMOVED: CSV saving code
    # df = pd.DataFrame(history_list)
    # df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # KEPT: Model checkpoint saving
    if val_acc > self.best_val_acc:
        self.best_val_acc = val_acc
        if output_dir:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
    
    return {'history': history_list}
```

### **2. ClassificationEvaluator.evaluate() Method**

**Original behavior:**
- Saves JSON file: `evaluation_metrics.json`
- Saves TXT file: `classification_report.txt`
- Saves PNG file: `confusion_matrix.png`

**Modified behavior:**
- ❌ Remove all file saving
- ✅ Return metrics dict with confusion_matrix as numpy array

**Implementation:**
```python
def evaluate(self, model, test_loader, output_dir=None):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        output_dir: IGNORED (no file saving)
    
    Returns:
        dict: Evaluation metrics including confusion_matrix as numpy array
    """
    # ... existing evaluation code ...
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm_array,  # numpy array
        'per_class_metrics': per_class_metrics
    }
    
    # REMOVED: All file saving code
    # with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
    #     json.dump(metrics, f, indent=2)
    # 
    # with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    #     f.write(report_str)
    #
    # plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    return metrics
```

### **3. Removed Methods**

The following methods should be **removed** from ClassificationEvaluator:

- `plot_training_curves()` → Will be implemented directly in experiment notebooks
- `generate_experiment_summary()` → Replaced with print statements in experiments

---

## ⚠️ Important Notes

### **Module Loading**
Each experiment notebook MUST start with:
```python
%run ./ResNet50_modules.ipynb
```

This ensures all classes, functions, and configurations are available.

### **Path Management**
- Use relative paths for data: `data/{STUDENT_ID}/Image_Classification/split_dataset`
- Use relative paths for output: `outputs/classification_[experiment]/`
- Ensure directories are created with `mkdir(parents=True, exist_ok=True)`

### **Inline Display**
- All matplotlib figures must call `plt.show()`
- Use `%matplotlib inline` in ResNet50_modules.ipynb
- All text output via `print()` statements

### **Model Saving**
- Only save `best_model.pth` during training
- Do NOT save `final_model.pth`
- Do NOT save any other files

---

## ✅ Completion Checklist

- [ ] Step 1: Read all source code files
- [ ] Step 2: Create ResNet50_modules.ipynb with all modifications
- [ ] Step 3: Create classification_ResNet50_baseline.ipynb
- [ ] Step 4: Create classification_ResNet50_reduced_v1.ipynb
- [ ] Step 5: Create classification_ResNet50_deeper_v1.ipynb
- [ ] Verify all notebooks can run independently (after loading modules)
- [ ] Verify all outputs display inline
- [ ] Verify only best_model.pth is saved
- [ ] Test one complete experiment flow

---

## 🔄 Recovery Instructions

If execution is interrupted:

1. Read this file: `notebooks/classification_ResNet50/make_notebook.md`
2. Check which steps are completed
3. Continue from the next incomplete step
4. Confirm with user after each step completion

---

**Last Updated:** 2026-05-06  
**Status:** Ready to execute Step 1
