# Experiment: ResNet50 Customized V6

**Date:** 2026-05-03 07:43:23

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
- **Add Conv After:** layer2

## Training Configuration

**Description:** V6 RUN-5: Lighter regularization (WD=1e-4) to recover from over-regularization (-2.01% drop)

## Results

- Best Val Accuracy: 0.9740
- Test Accuracy: 0.9679
- Test F1 (macro): 0.9674
- Test Precision (weighted): 0.9689
- Test Recall (weighted): 0.9679

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9977) and validation accuracy (0.9567) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

