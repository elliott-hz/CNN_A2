# Experiment: ResNet50 Baseline

**Date:** 2026-05-03 06:23:40

## Methodology

- **Architecture:** Standard ResNet50 with single FC layer (2048 → 10)
- **Training Strategy:** Single-phase training with ALL layers trainable
- **NO Layer Freezing:** Following teacher's requirement for correct methodology
- **Epochs:** 200
- **Learning Rate:** 0.001
- **Weight Decay:** 0.001
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
- **Modify Backbone:** False

## Training Configuration

**Description:** Baseline training with optimal hyperparameters (LR=1e-3, WD=1e-3, LS=0.10) from grid search

## Results

- Best Val Accuracy: 0.9697
- Test Accuracy: 0.9398
- Test F1 (macro): 0.9400
- Test Precision (weighted): 0.9438
- Test Recall (weighted): 0.9398

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9986) and validation accuracy (0.9567) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

1. **No Layer Freezing:** All layers trainable from start to ensure correct methodology
2. **Lower Learning Rate:** Started with 1e-4 since all layers are training
3. **Single-Phase Training:** Simplified training process while maintaining effectiveness
4. **Consistent Dataset Split:** Same split used across all classification experiments
