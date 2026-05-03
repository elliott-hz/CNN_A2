# Experiment: ResNet50 Customized V7

**Date:** 2026-05-02 16:32:09

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
- **Add Conv After:** layer3

## Training Configuration

**Description:** Added conv blocks after layer3 with single FC head, higher LR (1e-3), extended warmup (10 epochs)

## Results

- Best Val Accuracy: 0.9740
- Test Accuracy: 0.9639
- Test F1 (macro): 0.9634
- Test Precision (weighted): 0.9648
- Test Recall (weighted): 0.9639

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9984) and validation accuracy (0.9584) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

