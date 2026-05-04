# Training Configuration Refactoring - Complete Centralization

**Date:** 2026-05-01  
**Student ID:** 25509225  
**Purpose:** Centralize ALL training hyperparameters into TrainingConfig objects for clarity and reproducibility

---

## 🎯 Problem Solved

**Before:** Training parameters were scattered across multiple locations:
- `__init__` method: optimizer setup
- `train` method: scheduler, patience, early-stopping
- `TRAINING_CONFIGS` dict: basic hyperparameters

**After:** ALL training parameters are in **TrainingConfig dataclass** with 4 independent configurations.

---

## 📊 New Architecture

### **1. TrainingConfig Dataclass** (`src/training/ResNet50_trainer.py`)

```python
@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_type: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    
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

**Benefits:**
- ✅ All parameters in ONE place
- ✅ Type hints for IDE support
- ✅ Default values for optional parameters
- ✅ Clear documentation via docstrings
- ✅ Easy to serialize/deserialize

---

### **2. Four Independent Configurations**

#### **TRAINING_CONFIG_BASELINE**
```python
TRAINING_CONFIG_BASELINE = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    optimizer_type='adamw',
    epochs=50,
    use_scheduler=False,
    use_early_stopping=True,
    early_stopping_patience=10,
    label_smoothing=0.1,
    use_amp=True,
    description='Baseline training with moderate regularization'
)
```

#### **TRAINING_CONFIG_V1**
```python
TRAINING_CONFIG_V1 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=5e-3,              # ↑ Stronger regularization
    optimizer_type='adamw',
    epochs=60,                      # ↑ More epochs
    use_scheduler=False,
    use_early_stopping=True,
    early_stopping_patience=12,     # ↑ Longer patience
    label_smoothing=0.15,           # ↑ Higher smoothing
    use_amp=True,
    description='Enhanced FC head with stronger regularization (higher weight decay & label smoothing)'
)
```

#### **TRAINING_CONFIG_V2**
```python
TRAINING_CONFIG_V2 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=5e-3,
    optimizer_type='adamw',
    epochs=60,
    use_scheduler=False,
    use_early_stopping=True,
    early_stopping_patience=12,
    label_smoothing=0.15,
    use_amp=True,
    description='CNN backbone modification (add conv blocks) with enhanced FC head and strong regularization'
)
```

#### **TRAINING_CONFIG_V3**
```python
TRAINING_CONFIG_V3 = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-4,
    optimizer_type='adamw',
    epochs=50,
    use_scheduler=False,
    use_early_stopping=True,
    early_stopping_patience=10,
    label_smoothing=0.1,
    use_amp=True,
    description='Reduced depth backbone (remove layer3) with standard regularization'
)
```

---

### **3. Simplified Trainer API**

**Before:**
```python
trainer = ClassificationTrainer(
    model, 
    learning_rate=1e-4, 
    weight_decay=1e-4
)

history = trainer.train(
    train_loader, val_loader, criterion, 
    epochs=50,
    output_dir='...',
    patience=10
)
```

**After:**
```python
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)

history = trainer.train(
    train_loader, val_loader, criterion,
    output_dir='...'
)
```

**Key Changes:**
- ❌ Removed `learning_rate`, `weight_decay` from `__init__`
- ❌ Removed `epochs`, `patience`, `scheduler` from `train()` method
- ✅ All parameters come from `TrainingConfig` object
- ✅ Cleaner API, fewer arguments to pass around

---

### **4. Updated Experiment Scripts**

All 4 experiment scripts now follow the same simple pattern:

```
from src.training.ResNet50_trainer import (
    ClassificationTrainer, 
    TRAINING_CONFIG_BASELINE  # or V1, V2, V3
)

# Initialize trainer with complete config
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)

# Create loss function using config
criterion = torch.nn.CrossEntropyLoss(
    label_smoothing=TRAINING_CONFIG_BASELINE.label_smoothing
)

# Train (no need to pass epochs, patience, etc.)
history = trainer.train(
    train_loader, val_loader, criterion,
    str(output_dir / 'training')
)
```

---

## 🔧 Implementation Details

### **Trainer Initialization**

The `__init__` method now:
1. Accepts a `TrainingConfig` object
2. Sets up optimizer based on `config.optimizer_type`
3. Configures mixed precision based on `config.use_amp`
4. Creates scheduler if `config.use_scheduler` is True
5. Stores all config for later use

```python
def __init__(self, model: nn.Module, config: TrainingConfig):
    self.model = model
    self.config = config  # Store entire config
    
    # Setup optimizer based on config
    if config.optimizer_type == 'adamw':
        self.optimizer = torch.optim.AdamW(...)
    elif config.optimizer_type == 'adam':
        self.optimizer = torch.optim.Adam(...)
    # ... etc
    
    # Setup mixed precision
    self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    # Setup scheduler if enabled
    if config.use_scheduler:
        if config.scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(...)
        # ... etc
```

### **Training Loop**

The `train()` method now:
1. Uses `self.config.epochs` instead of parameter
2. Uses `self.config.use_early_stopping` for conditional logic
3. Uses `self.config.early_stopping_patience` for stopping
4. Prints full configuration at start for transparency

```python
def train(self, train_loader, val_loader, criterion, output_dir):
    epochs = self.config.epochs  # From config
    
    # Print configuration
    print(f'Training Configuration:')
    print(f'  - Epochs: {self.config.epochs}')
    print(f'  - Learning Rate: {self.config.learning_rate}')
    print(f'  - Weight Decay: {self.config.weight_decay}')
    # ... etc
    
    for epoch in range(epochs):
        # ... training loop
        
        # Early stopping check uses config
        if self.config.use_early_stopping and \
           early_stop_counter >= self.config.early_stopping_patience:
            break
```

---

## 📁 File Structure

```
src/
└── training/
    └── ResNet50_trainer.py
        ├── TrainingConfig (dataclass)
        ├── TRAINING_CONFIG_BASELINE
        ├── TRAINING_CONFIG_V1
        ├── TRAINING_CONFIG_V2
        ├── TRAINING_CONFIG_V3
        └── ClassificationTrainer class
            ├── __init__(model, config)
```

---

## ✅ Benefits of This Approach

### **1. Single Source of Truth**
All training hyperparameters are defined in ONE place per experiment. No more hunting through code to find where parameters are set.

### **2. Type Safety**
Using `@dataclass` provides:
- Type hints for IDE autocomplete
- Runtime type checking (if needed)
- Clear documentation of expected types

### **3. Reproducibility**
Each `TrainingConfig` can be:
- Printed for logging
- Serialized to JSON/YAML
- Saved with model checkpoints
- Easily shared between team members

### **4. Flexibility**
Easy to add new parameters:
```python
# Just add to dataclass
@dataclass
class TrainingConfig:
    # ... existing fields
    gradient_accumulation_steps: int = 1  # NEW!
```

### **5. Clean Experiment Scripts**
Experiment scripts only need to:
1. Import the right config
2. Pass it to trainer
3. That's it!

### **6. Easy Comparison**
To compare experiments, just diff the configs:
```python
print(TRAINING_CONFIG_BASELINE)
print(TRAINING_CONFIG_V2)
# See exactly what's different
```

---

## 🚀 Usage Examples

### **Run Baseline Experiment**
```bash
python experiments/classification_ResNet50_baseline.py
```
Uses: `BASELINE_CONFIG` + `TRAINING_CONFIG_BASELINE`

### **Run Customized V2 Experiment**
```bash
python experiments/classification_ResNet50_v2.py
```
Uses: `CUSTOMIZED_V2_CONFIG` + `TRAINING_CONFIG_V2`

### **Create Custom Configuration**
```python
from src.training.ResNet50_trainer import TrainingConfig

my_config = TrainingConfig(
    learning_rate=5e-5,
    weight_decay=1e-3,
    epochs=100,
    use_scheduler=True,
    scheduler_type='cosine',
    description='My custom experiment'
)

trainer = ClassificationTrainer(model, config=my_config)
```

---

## 📊 Configuration Comparison Table

| Parameter | BASELINE | V1 | V2 | V3 |
|-----------|----------|----|----|----|
| learning_rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| weight_decay | 1e-4 | 5e-3 | 5e-3 | 1e-4 |
| optimizer_type | adamw | adamw | adamw | adamw |
| epochs | 50 | 60 | 60 | 50 |
| use_scheduler | False | False | False | False |
| use_early_stopping | True | True | True | True |
| early_stopping_patience | 10 | 12 | 12 | 10 |
| label_smoothing | 0.1 | 0.15 | 0.15 | 0.1 |
| use_amp | True | True | True | True |
| **Use Case** | Control | FC enhancement | CNN mod + FC | Reduced depth |

---

## 💡 Best Practices

### **1. Always Use Named Configs**
```python
# ✅ Good
trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)

# ❌ Bad
trainer = ClassificationTrainer(model, learning_rate=1e-4, ...)
```

### **2. Document Changes**
When modifying a config, update the `description` field:
```python
TRAINING_CONFIG_V2 = TrainingConfig(
    # ... params
    description='Updated: Added gradient accumulation for larger batches'
)
```

### **3. Version Control Configs**
Keep configs in version control so you can track changes over time.

### **4. Log Full Config**
The trainer already prints the full config at training start. Keep this for reproducibility.

---

## 🎯 Summary

**What Changed:**
- ✅ Created `TrainingConfig` dataclass
- ✅ Defined 4 independent configs (BASELINE, V1, V2, V3)
- ✅ Moved ALL training parameters into configs
- ✅ Simplified `ClassificationTrainer` API
- ✅ Updated all 4 experiment scripts
- ✅ Updated evaluator to handle TrainingConfig

**Result:**
- Cleaner, more maintainable code
- All hyperparameters in one place
- Easy to compare experiments
- Better reproducibility
- Follows user's preference for centralized configuration management

---

**All training configuration is now properly centralized and organized!** 🎉
