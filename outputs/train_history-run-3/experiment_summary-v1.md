# Experiment: ResNet50 Customized V1

**Date:** 2026-05-02 16:34:40

## Methodology

- **Architecture:** ResNet50 with enhanced multi-layer FC head (2048 → 512 → 256 → 10) and BatchNorm
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
- **FC Hidden Dimensions:** [256]
- **Use BatchNorm:** True
- **Modify Backbone:** False

## Training Configuration

**Description:** Enhanced FC head with higher LR (1e-3), extended warmup (10 epochs) and dynamic LR reduction

## Results

- Best Val Accuracy: 0.9740
- Test Accuracy: 0.9679
- Test F1 (macro): 0.9672
- Test Precision (weighted): 0.9684
- Test Recall (weighted): 0.9679

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9955) and validation accuracy (0.9636) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

1. **Enhanced FC Head:** Multi-layer classifier enables complex feature combinations
2. **Batch Normalization:** Stabilizes training in deeper FC layers
3. **Stronger Regularization:** Higher dropout (0.7), weight decay (5e-3), and label smoothing (0.15)
4. **No Backbone Modification:** Kept standard ResNet50 backbone for comparison
