# Experiment: ResNet50 Customized V4

**Date:** 2026-05-02 16:23:49

## Methodology

- **Architecture:** Custom ResNet50 architecture
- **Training Strategy:** Single-phase training with ALL layers trainable
- **NO Layer Freezing:** Following teacher's requirement for correct methodology
- **Epochs:** 200
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Optimizer:** ADAMW
- **Label Smoothing:** 0.1
- **Early Stopping:** Enabled (patience=50)
- **Scheduler:** Enabled
- **Mixed Precision:** Enabled

## Model Configuration Details

- **Num Classes:** 10
- **Dropout Rate:** 0.5
- **Pretrained:** False
- **FC Architecture:** Single Layer (Baseline style)
- **Use BatchNorm:** True
- **Modify Backbone:** True
- **Remove Layer:** layer4

## Training Configuration

**Description:** Reduced depth backbone (layer4 removed) with higher LR (1e-3), extended warmup (10 epochs) and dynamic LR reduction

## Results

- Best Val Accuracy: 0.9610
- Test Accuracy: 0.9357
- Test F1 (macro): 0.9343
- Test Precision (weighted): 0.9389
- Test Recall (weighted): 0.9357

## Overfitting/Underfitting Analysis

**Pattern Detected:** overfitting

Model shows signs of overfitting. Training accuracy (0.9942) is significantly higher than validation accuracy (0.9160). Validation loss may be increasing.

**Recommendation:** Consider: 1) Add dropout, 2) Increase weight decay, 3) Add data augmentation, 4) Use early stopping, 5) Reduce model complexity

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

