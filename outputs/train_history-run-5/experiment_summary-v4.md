# Experiment: ResNet50 Customized V4

**Date:** 2026-05-03 07:46:09

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

**Description:** V4 RUN-5: Keep maximum regularization (fix was successful, no changes needed)

## Results

- Best Val Accuracy: 0.9697
- Test Accuracy: 0.9478
- Test F1 (macro): 0.9476
- Test Precision (weighted): 0.9577
- Test Recall (weighted): 0.9478

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9922) and validation accuracy (0.9584) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

