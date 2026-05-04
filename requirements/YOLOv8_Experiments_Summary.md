# YOLOv8 Detection Experiments - Implementation Summary

**Date:** 2026-05-04  
**Last Updated:** 2026-05-04 (Architecture Fix)
**Student ID:** 25509225  
**Course:** 42028 Deep Learning and Convolutional Neural Networks

---

## 📋 Overview

Three YOLOv8 detection experiments have been implemented with **true CNN customization** (adding/removing convolutional layers in backbone) as required by the assignment specification.

**Architecture Modifications:**
- ✅ **V2:** Adds 2 Conv layers in backbone after layer2
- ✅ **V3:** Removes 6 Conv layers by reducing layer4 C2f repeats
- ✅ All modifications are in **Backbone only** (not Neck)
- ✅ Compliant with teacher's requirement for true CNN customization

---

## 🎯 Experiment Designs

### **V1: Baseline (detection_YOLOv8_v1.py)**

**Purpose:** Control group for comparison

**Configuration:**
- Model: Standard YOLOv8m
- Input size: 640×640
- Epochs: 100
- Batch size: 16 (optimized for T4 GPU)
- Learning rate: 0.001
- Optimizer: Adam
- Weight decay: 1e-4
- Early stopping patience: 15

**Customization:** None (baseline)

**Expected Parameters:** ~25.9M

---

### **V2: Deeper Backbone (detection_YOLOv8_v2.py)**

**Purpose:** Test if adding convolutional layers in shallow backbone improves feature extraction

**Configuration:**
- Model: Custom YOLOv8m with deeper backbone
- Input size: 640×640
- Epochs: 120 (more epochs for deeper model)
- Batch size: 12 (reduced due to larger model)
- Learning rate: 0.0005 (lower for stability)
- Optimizer: Adam
- Weight decay: 5e-4 (stronger regularization)
- Early stopping patience: 20
- LR schedule: Cosine annealing

**Customization Details:**
- **Type:** Added Convolutional Layers in Backbone
- **Location:** After layer2 (C2f module at P2/4 level)
- **Changes:**
  1. Added Conv(128, 3×3, stride=1, padding=1) - maintains resolution
  2. Added Conv(128, 3×3, stride=1, padding=1) - deepens feature representation
- **Total added:** 2 convolutional layers
- **Parameter increase:** ~0.5M (~26.4M total)
- **All subsequent indices shifted by +2**

**Custom YAML:** `src/models/yolov8m_custom_deeper.yaml`

**Hypothesis:** Deeper shallow layers should capture more fine-grained features for solar panel damage detection.

---

### **V3: Shallower Backbone (detection_YOLOv8_v3.py)**

**Purpose:** Test if a lighter model can maintain performance with faster inference

**Configuration:**
- Model: Custom YOLOv8m with shallower backbone
- Input size: 640×640
- Epochs: 80 (fewer epochs for simpler model)
- Batch size: 20 (increased due to smaller model)
- Learning rate: 0.001
- Optimizer: Adam
- Weight decay: 1e-4
- Early stopping patience: 12

**Customization Details:**
- **Type:** Removed Convolutional Layers from Backbone
- **Location:** Layer4 (P4/16 level)
- **Changes:**
  1. Reduced layer4 C2f repeats: 6 → 3 (removed 3 bottlenecks)
  2. Each bottleneck contains 2 conv layers, so removed 6 conv layers total
- **Total removed:** 6 convolutional layers
- **Parameter decrease:** ~3M (~22.9M total)
- **All layer indices remain unchanged** (stable architecture)

**Custom YAML:** `src/models/yolov8m_custom_shallow.yaml`

**Hypothesis:** Lighter model should enable faster inference and larger batch sizes, potentially reducing overfitting while maintaining reasonable accuracy.

---

## 📊 Comparison Table

| Experiment | Script | Model Type | Backbone Changes | Parameters | Epochs | Batch Size | LR | Weight Decay |
|------------|--------|-----------|-------------------|------------|--------|------------|-----|--------------|
| **V1** | detection_YOLOv8_v1.py | Standard YOLOv8m | 0 (Baseline) | ~25.9M | 100 | 16 | 0.001 | 1e-4 |
| **V2** | detection_YOLOv8_v2.py | Deeper Backbone | **+2 Conv layers** (after layer2) | ~26.4M | 120 | 12 | 0.0005 | 5e-4 |
| **V3** | detection_YOLOv8_v3.py | Shallower Backbone | **-6 Conv layers** (layer4 reduced) | ~22.9M | 80 | 20 | 0.001 | 1e-4 |

---

## 🔧 Technical Implementation

### **1. Dataset Configuration**

All experiments use the same dataset path:
```python
DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
```

This ensures:
- ✅ Consistent data splits across all experiments
- ✅ Fair comparison methodology
- ✅ No resplitting (as per teacher requirement)

### **2. Custom YAML Configurations**

Two custom YAML files created in `src/models/`:

**yolov8m_custom_deeper.yaml (V2):**
- Adds 2 Conv layers after backbone layer2 (C2f module)
- Maintains spatial resolution with stride=1, padding=1
- All subsequent layer indices shifted by +2
- Neck and Head references updated accordingly

**yolov8m_custom_shallow.yaml (V3):**
- Reduces layer4 C2f repeats from 6 to 3
- Removes 3 bottleneck blocks (6 conv layers total)
- Keeps all layer indices stable (no structural changes)
- Safer approach than deleting entire layers

### **3. Model Loading**

Updated `YOLOv8Detector` class supports both standard and custom models:

```python
# Standard model (V1)
model = YOLOv8Detector(**YOLOV8_BASELINE_CONFIG)

# Custom model from YAML (V2, V3)
model = YOLOv8Detector(**YOLOV8_V2_CONFIG)  # or V3_CONFIG
```

When `model_yaml` is provided:
1. Creates model from custom YAML
2. Loads pretrained weights from standard YOLOv8m
3. Fine-tunes all layers (no freezing!)

### **4. Training Features**

All experiments include:
- ✅ Mixed precision training (AMP) for memory efficiency
- ✅ Early stopping to prevent overfitting
- ✅ Automatic checkpoint saving (best model)
- ✅ CSV metrics logging (every epoch)
- ✅ Comprehensive experiment summaries (Markdown)

---

## 📁 File Structure

```
CNN_A2/
├── experiments/
│   ├── detection_YOLOv8_v1.py          # Baseline experiment
│   ├── detection_YOLOv8_v2.py          # Deeper backbone (+6 conv layers)
│   └── detection_YOLOv8_v3.py          # Shallower backbone (-8 conv layers)
│
├── src/models/
│   ├── YOLOv8DetectorModel.py          # Updated with custom YAML support
│   ├── yolov8m_custom_deeper.yaml      # V2 architecture config
│   └── yolov8m_custom_shallow.yaml     # V3 architecture config
│
└── outputs/
    ├── detection_yolov8_v1/run_TIMESTAMP/
    │   ├── training/
    │   │   ├── train/                  # Ultralytics outputs
    │   │   │   └── results.csv         # Epoch-by-epoch metrics
    │   │   └── training_history.csv    # Copied for easy access
    │   ├── evaluation/
    │   │   └── evaluation_metrics.json
    │   └── experiment_summary.md
    │
    ├── detection_yolov8_v2/run_TIMESTAMP/
    │   └── ... (same structure)
    │
    └── detection_yolov8_v3/run_TIMESTAMP/
        └── ... (same structure)
```

---

## ✅ Compliance with Assignment Requirements

| Requirement | Status | Implementation |
|------------|--------|----------------|
| **True CNN Customization** | ✅ | V2 adds 6 conv layers, V3 removes 8 conv layers |
| **No Layer Freezing** | ✅ | All layers trainable from epoch 1 |
| **Consistent Data Splits** | ✅ | Same yolo/data.yaml used across all experiments |
| **Methodology Focus** | ✅ | Fair comparison with controlled variables |
| **Training Curves** | ✅ | CSV logging enables curve plotting |
| **Comprehensive Documentation** | ✅ | Markdown summaries with detailed analysis |

---

## 🚀 How to Run

### **Run All Three Experiments:**

```bash
cd /Users/elliott/vscode_workplace/CNN_A2

# V1: Baseline
python experiments/detection_YOLOv8_v1.py

# V2: Deeper Backbone
python experiments/detection_YOLOv8_v2.py

# V3: Shallower Backbone
python experiments/detection_YOLOv8_v3.py
```

### **Expected Runtime (T4 GPU):**
- V1: ~2-3 hours (100 epochs)
- V2: ~3-4 hours (120 epochs, larger model)
- V3: ~1.5-2 hours (80 epochs, smaller model)

---

## 📈 Expected Outcomes

### **Performance Comparison:**

**V1 (Baseline):**
- Reference point for mAP, precision, recall
- Balanced speed vs accuracy

**V2 (Deeper Backbone):**
- Potentially better shallow feature extraction
- May improve detection of small damage patterns
- Slightly higher computational cost (~0.5M more params)
- Should show improved early-layer feature learning

**V3 (Shallower Backbone):**
- Significantly faster training and inference
- Lower risk of overfitting with fewer parameters
- Can use larger batch size (20 vs 16)
- May maintain reasonable performance despite reduced depth

### **Analysis Points:**

1. **Does deeper shallow backbone improve accuracy?**
   - Compare V2 vs V1 mAP scores
   - Check if additional conv layers help with fine-grained features
   - Analyze if small damage detection improves

2. **Can lighter model maintain reasonable performance?**
   - Compare V3 vs V1 trade-offs
   - Evaluate speed vs accuracy balance
   - Check if reduced depth affects multi-scale detection

3. **Overfitting analysis:**
   - Monitor train/val loss curves
   - Check early stopping triggers
   - Analyze gap between training and validation metrics
   - V3 should show less overfitting tendency

---

## 🔍 Post-Experiment Analysis

After running all experiments:

1. **Compare CSV metrics:**
   ```python
   import pandas as pd
   
   v1 = pd.read_csv('outputs/detection_yolov8_v1/run_XXX/training/training_history.csv')
   v2 = pd.read_csv('outputs/detection_yolov8_v2/run_XXX/training/training_history.csv')
   v3 = pd.read_csv('outputs/detection_yolov8_v3/run_XXX/training/training_history.csv')
   
   # Plot comparison curves
   ```

2. **Analyze confusion matrices:**
   - Check which classes benefit from deeper architecture
   - Identify misclassification patterns

3. **Write comprehensive report:**
   - Include training curves for all 3 experiments
   - Discuss methodology and design decisions
   - Analyze overfitting/underfitting patterns
   - Compare parameter counts vs performance

---

## ⚠️ Important Notes

1. **GPU Memory Management:**
   - V2 uses smaller batch size (12) due to larger model
   - If OOM errors occur, reduce batch_size further
   - Mixed precision (AMP) is enabled by default

2. **Training Time:**
   - V2 will take longest (deeper model + more epochs)
   - V3 will be fastest (lighter model + fewer epochs)
   - Monitor progress via terminal output

3. **Reproducibility:**
   - All configurations are saved in experiment summaries
   - Custom YAML files are version-controlled
   - Results are timestamped for tracking

4. **Teacher's Emphasis:**
   - Focus on **methodology correctness**, not just accuracy
   - Document assumptions and design decisions
   - Ensure fair comparisons across experiments

---

## 📝 Next Steps

1. ✅ Code implementation complete
2. ⏳ Run experiments (user preference: batch execution after all setup)
3. ⏳ Analyze results and generate comparison plots
4. ⏳ Write assignment report with findings

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Last Updated:** 2026-05-04
