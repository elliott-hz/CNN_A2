# Layer Freezing/Unfreezing Guide for ResNet50Classifier

**Date:** 2026-05-01  
**Student ID:** 25509225  

---

## Overview

The `ResNet50Classifier` model supports optional layer freezing/unfreezing for flexible training strategies. While the current experiments use **NO layer freezing** (all layers trainable from epoch 1), the methods are available if you want to experiment with different training approaches.

---

## Available Methods

### 1. `freeze_backbone()`

Freezes all backbone parameters (layer1, layer2, layer3, layer4, bn1).

```python
model.freeze_backbone()
```

**Effect:**
- Only the classifier head (FC layers) will be trained
- Backbone weights remain unchanged
- Useful for initial training phase with limited data

### 2. `unfreeze_backbone(unfreeze_layer2=False)`

Unfreezes backbone layers for fine-tuning.

```python
# Option 1: Unfreeze only layer3 and layer4 (default)
model.unfreeze_backbone()

# Option 2: Also unfreeze layer2 for extended fine-tuning
model.unfreeze_backbone(unfreeze_layer2=True)
```

**Effect:**
- By default: Unfreezes layer3, layer4, and bn1
- With `unfreeze_layer2=True`: Also unfreezes layer2
- Allows deeper feature adaptation

---

## Usage Examples

### Example 1: Two-Phase Training (Traditional Approach)

```python
from src.models.ResNet50ClassifierModel import ResNet50Classifier, BASELINE_CONFIG
from src.training.classification_trainer import ClassificationTrainer, TRAINING_CONFIG_BASELINE

# Initialize model
model = ResNet50Classifier(**BASELINE_CONFIG)

# ========== Phase 1: Train only FC layers ==========
print("Phase 1: Training with frozen backbone...")
model.freeze_backbone()

trainer1 = ClassificationTrainer(model, config=TRAINING_CONFIG_BASELINE)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

history1 = trainer1.train(
    train_loader, val_loader, criterion,
    output_dir='outputs/phase1'
)

# ========== Phase 2: Fine-tune entire network ==========
print("\nPhase 2: Fine-tuning with unfrozen backbone...")
model.unfreeze_backbone(unfreeze_layer2=False)  # or True for extended

# Create new trainer with lower learning rate for fine-tuning
from src.training.classification_trainer import TrainingConfig
fine_tune_config = TrainingConfig(
    learning_rate=1e-5,  # Lower LR for fine-tuning
    weight_decay=1e-4,
    epochs=50,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=5,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=15,
    label_smoothing=0.1,
    use_amp=True,
    description='Fine-tuning phase with lower learning rate'
)

trainer2 = ClassificationTrainer(model, config=fine_tune_config)
history2 = trainer2.train(
    train_loader, val_loader, criterion,
    output_dir='outputs/phase2'
)
```

### Example 2: Progressive Unfreezing (Gradual Approach)

```python
model = ResNet50Classifier(**BASELINE_CONFIG)

# Step 1: Freeze everything except classifier
model.freeze_backbone()
trainer1 = ClassificationTrainer(model, config=config_phase1)
trainer1.train(train_loader, val_loader, criterion, 'outputs/step1')

# Step 2: Unfreeze top layers (layer4)
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
trainer2 = ClassificationTrainer(model, config=config_phase2)
trainer2.train(train_loader, val_loader, criterion, 'outputs/step2')

# Step 3: Unfreeze more layers (layer3 + layer4)
model.unfreeze_backbone(unfreeze_layer2=False)
trainer3 = ClassificationTrainer(model, config=config_phase3)
trainer3.train(train_loader, val_loader, criterion, 'outputs/step3')

# Step 4: Unfreeze even more (layer2 + layer3 + layer4)
model.unfreeze_backbone(unfreeze_layer2=True)
trainer4 = ClassificationTrainer(model, config=config_phase4)
trainer4.train(train_loader, val_loader, criterion, 'outputs/step4')
```

### Example 3: Check Which Layers Are Trainable

```python
def print_trainable_layers(model):
    """Print which layers are currently trainable."""
    print("\nTrainable Layers:")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✓ {name}: {param.shape}")
        else:
            print(f"✗ {name}: {param.shape} (frozen)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 60)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Frozen params: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")

# Usage
model = ResNet50Classifier(**BASELINE_CONFIG)
print_trainable_layers(model)  # All layers trainable

model.freeze_backbone()
print_trainable_layers(model)  # Only classifier trainable

model.unfreeze_backbone()
print_trainable_layers(model)  # layer3, layer4, bn1, and classifier trainable
```

---

## Current Experiments Configuration

All 5 current experiments use **NO layer freezing**:

| Experiment | Freezing Strategy | Rationale |
|------------|------------------|-----------|
| Baseline | None (all trainable) | Teacher's requirement: NO layer freezing |
| V1 | None (all trainable) | Teacher's requirement: NO layer freezing |
| V2 | None (all trainable) | Teacher's requirement: NO layer freezing |
| V3 | None (all trainable) | Teacher's requirement: NO layer freezing |
| V4 | None (all trainable) | Teacher's requirement: NO layer freezing |

**Why?**
- Teacher explicitly requires: "Both classification experiments must NOT freeze any layers"
- All layers trainable from epoch 1 allows end-to-end optimization
- Simplifies training pipeline (single phase instead of multi-phase)

---

## When to Use Layer Freezing?

### ✅ Good Scenarios for Freezing

1. **Very small dataset** (< 1000 images per class)
   - Prevents overfitting by keeping pretrained features
   - Only trains task-specific classifier

2. **Limited computational resources**
   - Faster training (fewer gradients to compute)
   - Less memory usage

3. **Transfer learning from similar domain**
   - Pretrained features already very relevant
   - Only need minor adjustments

### ❌ When NOT to Freeze

1. **Sufficient data available** (> 100 images per class)
   - Can afford to fine-tune entire network
   - Better performance potential

2. **Domain is very different from ImageNet**
   - Pretrained features may not be optimal
   - Need to learn domain-specific features

3. **Teacher/assignment requirements**
   - Some methodologies require full fine-tuning
   - Follow assignment specifications

---

## Parameter Recommendations

### For Two-Phase Training

```python
# Phase 1: Frozen backbone (train classifier only)
config_phase1 = TrainingConfig(
    learning_rate=1e-3,      # Higher LR for classifier
    weight_decay=1e-4,
    epochs=10-20,            # Short phase
    use_early_stopping=True,
    early_stopping_patience=5,
    # ... other params
)

# Phase 2: Unfrozen backbone (fine-tune everything)
config_phase2 = TrainingConfig(
    learning_rate=1e-5,      # Much lower LR for fine-tuning
    weight_decay=1e-4,
    epochs=30-50,            # Longer phase
    use_early_stopping=True,
    early_stopping_patience=10,
    # ... other params
)
```

### Learning Rate Guidelines

| Phase | Layers Trained | Recommended LR | Reason |
|-------|---------------|----------------|--------|
| Frozen backbone | Classifier only | 1e-3 to 1e-4 | Classifier needs to learn from scratch |
| Unfrozen (layer3+4) | Top layers + classifier | 1e-5 to 1e-6 | Small adjustments to pretrained features |
| Unfrozen (layer2+3+4) | Most backbone + classifier | 1e-5 to 1e-6 | Moderate adjustments |
| All layers | Everything | 1e-5 | Full fine-tuning |

---

## Comparison: Freezing vs No Freezing

### With Freezing (Two-Phase)

**Pros:**
- ✅ Faster initial training
- ✅ Less risk of overfitting on small datasets
- ✅ Preserves pretrained knowledge
- ✅ Good for transfer learning from similar domains

**Cons:**
- ❌ More complex training pipeline
- ❌ May not achieve best performance
- ❌ Requires careful LR scheduling between phases
- ❌ Violates teacher's current requirement

### Without Freezing (Single-Phase)

**Pros:**
- ✅ Simpler training pipeline
- ✅ Potentially better performance (full optimization)
- ✅ End-to-end learning
- ✅ Complies with teacher's requirement

**Cons:**
- ❌ Slower training (more gradients)
- ❌ Higher risk of overfitting on small datasets
- ❌ Needs more data or stronger regularization

---

## Quick Reference

```python
# Check current state
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Freeze backbone
model.freeze_backbone()

# Unfreeze (default: layer3 + layer4 + bn1)
model.unfreeze_backbone()

# Unfreeze more (layer2 + layer3 + layer4 + bn1)
model.unfreeze_backbone(unfreeze_layer2=True)

# Manual control (if needed)
for param in model.backbone.layer2.parameters():
    param.requires_grad = True  # or False
```

---

## Summary

- ✅ Methods are available: `freeze_backbone()` and `unfreeze_backbone()`
- ✅ Current experiments don't use them (per teacher's requirement)
- ✅ You can enable them if you want to experiment with two-phase training
- ✅ Remember to adjust learning rates between phases
- ✅ Always check which layers are trainable before training

---

**Status:** Methods preserved and documented for future use 🎉
