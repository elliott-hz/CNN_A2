# Experiment: ResNet50 Customized V2

**Date:** 2026-05-02 16:33:33

## Methodology

- **Architecture:** ResNet50 with backbone modification (added conv blocks after layer2) and enhanced multi-layer FC head (2048 → 512 → 256 → 10)
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
- **FC Hidden Dimensions:** [512, 256]
- **Use BatchNorm:** True
- **Modify Backbone:** True
- **Add Conv After:** layer2

## Training Configuration

**Description:** CNN backbone modification with higher LR (1e-3), extended warmup (10 epochs) and dynamic LR reduction

## Results

- Best Val Accuracy: 0.9697
- Test Accuracy: 0.9598
- Test F1 (macro): 0.9591
- Test Precision (weighted): 0.9611
- Test Recall (weighted): 0.9598

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9832) and validation accuracy (0.9654) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

1. **Backbone Structural Change:** Added convolutional blocks after layer2 to increase capacity
2. **Enhanced FC Head:** Multi-layer classifier with BatchNorm for stable training
3. **Balanced Regularization:** Moderate dropout (0.6) with strong weight decay (5e-3)
4. **TRUE CNN Customization:** Modifies actual CNN architecture, not just hyperparameters
5. **All Layers Trainable:** Ensures proper end-to-end learning
