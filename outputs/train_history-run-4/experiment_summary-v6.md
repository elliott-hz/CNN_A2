# Experiment: ResNet50 Customized V6

**Date:** 2026-05-03 06:25:57

## Methodology

- **Architecture:** Custom ResNet50 architecture
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
- **Add Conv After:** layer2

## Training Configuration

**Description:** Add conv after layer2 with optimal hyperparameters (LR=1e-3, WD=1e-3, LS=0.10)

## Results

- Best Val Accuracy: 0.9697
- Test Accuracy: 0.9438
- Test F1 (macro): 0.9436
- Test Precision (weighted): 0.9480
- Test Recall (weighted): 0.9438

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9982) and validation accuracy (0.9532) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

