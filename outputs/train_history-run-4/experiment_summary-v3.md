# Experiment: ResNet50 Customized V3

**Date:** 2026-05-03 06:28:13

## Methodology

- **Architecture:** ResNet50 with reduced depth (layer3 removed) and standard single FC layer (2048 → 10)
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
- **Modify Backbone:** True
- **Remove Layer:** layer3

## Training Configuration

**Description:** Remove layer3 with optimal hyperparameters (LR=1e-3, WD=1e-3, LS=0.10) from grid search

## Results

- Best Val Accuracy: 0.9827
- Test Accuracy: 0.9719
- Test F1 (macro): 0.9714
- Test Precision (weighted): 0.9725
- Test Recall (weighted): 0.9719

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (1.0000) and validation accuracy (0.9723) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

1. **Reduced Depth:** Removed layer3 to create lighter model and reduce overfitting risk
2. **Standard FC Head:** Simple single-layer classifier for fair comparison
3. **Moderate Regularization:** Standard dropout and weight decay settings
4. **Alternative Customization Strategy:** Demonstrates that removing layers is also valid customization
5. **All Layers Trainable:** Maintains correct methodology
