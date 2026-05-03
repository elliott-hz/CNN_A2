# Experiment: ResNet50 Customized V4

**Date:** 2026-05-03 06:24:57

## Methodology

- **Architecture:** Custom ResNet50 architecture
- **Training Strategy:** Single-phase training with ALL layers trainable
- **NO Layer Freezing:** Following teacher's requirement for correct methodology
- **Epochs:** 200
- **Learning Rate:** 0.0005
- **Weight Decay:** 0.005
- **Optimizer:** ADAMW
- **Label Smoothing:** 0.15
- **Early Stopping:** Enabled (patience=50)
- **Scheduler:** Enabled
- **Mixed Precision:** Enabled

## Model Configuration Details

- **Num Classes:** 10
- **Dropout Rate:** 0.7
- **Pretrained:** False
- **FC Architecture:** Single Layer (Baseline style)
- **Use BatchNorm:** True
- **Modify Backbone:** True
- **Remove Layer:** layer4

## Training Configuration

**Description:** V4 FIX ATTEMPT: Remove layer4 with maximum regularization (LR=5e-4, WD=5e-3, LS=0.15)

## Results

- Best Val Accuracy: 0.9740
- Test Accuracy: 0.9639
- Test F1 (macro): 0.9634
- Test Precision (weighted): 0.9660
- Test Recall (weighted): 0.9639

## Overfitting/Underfitting Analysis

**Pattern Detected:** overfitting

Model shows signs of overfitting. Training accuracy (0.9995) is significantly higher than validation accuracy (0.9541). Validation loss may be increasing.

**Recommendation:** Consider: 1) Add dropout, 2) Increase weight decay, 3) Add data augmentation, 4) Use early stopping, 5) Reduce model complexity

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

