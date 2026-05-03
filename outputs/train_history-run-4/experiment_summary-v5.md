# Experiment: ResNet50 Customized V5

**Date:** 2026-05-03 06:26:56

## Methodology

- **Architecture:** Custom ResNet50 architecture
- **Training Strategy:** Single-phase training with ALL layers trainable
- **NO Layer Freezing:** Following teacher's requirement for correct methodology
- **Epochs:** 200
- **Learning Rate:** 0.0005
- **Weight Decay:** 0.005
- **Optimizer:** ADAMW
- **Label Smoothing:** 0.05
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
- **Add Conv After:** layer1

## Training Configuration

**Description:** V5 MAXIMUM PERFORMANCE: Add conv after layer1 with optimal config (LR=5e-4, WD=5e-3, LS=0.05)

## Results

- Best Val Accuracy: 0.9740
- Test Accuracy: 0.9679
- Test F1 (macro): 0.9674
- Test Precision (weighted): 0.9709
- Test Recall (weighted): 0.9679

## Overfitting/Underfitting Analysis

**Pattern Detected:** good_fit

Model appears to have a good fit. Training accuracy (0.9986) and validation accuracy (0.9671) are reasonably close.

**Recommendation:** Model performance is acceptable. Continue monitoring.

## Training Curves

See `visualization/training_curves.png` for:
- Training vs Validation Loss
- Training vs Validation Accuracy

## Key Design Decisions

