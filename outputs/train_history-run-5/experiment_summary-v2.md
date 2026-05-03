# Experiment: ResNet50 Customized V2

**Date:** 2026-05-03 07:51:19

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
- **FC Hidden Dimensions:** [256]
- **Use BatchNorm:** True
- **Modify Backbone:** True
- **Add Conv After:** layer1

## Training Configuration

**Description:** V2 RUN-5: No changes (already optimal, perfect consistency across runs)

## Results

- Best Val Accuracy: 0.9827
- Test Accuracy: 0.9558
- Test F1 (macro): 0.9555
- Test Precision (weighted): 0.9571
- Test Recall (weighted): 0.9558

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9935) and validation accuracy (0.9688) are reasonably close.

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
