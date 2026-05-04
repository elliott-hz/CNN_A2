# YOLOv8 Detection Experiments - Implementation Summary

**Date:** 2026-05-04  
**Student ID:** 25509225  
**Course:** 42028 Deep Learning and Convolutional Neural Networks

---

## 📋 Overview

Three YOLOv8 detection experiments have been implemented with **true CNN customization** (adding/removing convolutional layers) as required by the assignment specification.

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

**Purpose:** Test if adding convolutional layers improves feature extraction

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
- **Type:** Added Convolutional Layers
- **Location:** Backbone (between layer3 and layer4)
- **Changes:**
  1. Added 1×1 Conv (1024→512 channels) - dimensionality reduction
  2. Added C2f module with 2 bottlenecks (4 conv layers) - feature enhancement
  3. Added 3×3 Conv (512→1024 channels) - dimensionality restoration
- **Total added:** 6 convolutional layers
- **Parameter increase:** ~15-18% (~29-30M total)

**Custom YAML:** `src/models/yolov8m_custom_deeper.yaml`

**Hypothesis:** Deeper backbone should capture more complex patterns in solar panel damage detection.

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
- Weight depth: 1e-4
- Early stopping patience: 12

**Customization Details:**
- **Type:** Removed Convolutional Layers
- **Location:** Backbone layer4 and layer5
- **Changes:**
  1. Reduced layer4 C2f: 6 repeats → 3 repeats (removed 6 conv layers)
  2. Reduced layer5 C2f: 3 repeats → 2 repeats (removed 2 conv layers)
- **Total removed:** 8 convolutional layers
- **Parameter decrease:** ~12-15% (~22-23M total)

**Custom YAML:** `src/models/yolov8m_custom_shallow.yaml`

**Hypothesis:** Lighter model should enable faster inference and larger batch sizes, potentially reducing overfitting.

---

## 📊 Comparison Table

| Experiment | Script | Model Type | Conv Layer Changes | Parameters | Epochs | Batch Size | LR | Weight Decay |
|------------|--------|-----------|-------------------|------------|--------|------------|-----|--------------|
| **V1** | detection_YOLOv8_v1.py | Standard YOLOv8m | 0 (Baseline) | ~25.9M | 100 | 16 | 0.001 | 1e-4 |
| **V2** | detection_YOLOv8_v2.py | Deeper Backbone | **+6 layers** | ~29-30M | 120 | 12 | 0.0005 | 5e-4 |
| **V3** | detection_YOLOv8_v3.py | Shallower Backbone | **-8 layers** | ~22-23M | 80 | 20 | 0.001 | 1e-4 |

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
- Adds extra C2f module in backbone
- Modifies neck connections to accommodate new layers
- Maintains compatibility with YOLOv8 architecture

**yolov8m_custom_shallow.yaml (V3):**
- Reduces C2f repetitions in layer4 and layer5
- Keeps neck and head unchanged
- Simpler architecture with fewer parameters

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

**V2 (Deeper):**
- Potentially higher mAP@0.5 and mAP@0.5:0.95
- Better detection of subtle damage patterns
- Higher computational cost
- May show signs of overfitting if not regularized properly

**V3 (Shallower):**
- Slightly lower mAP but faster inference
- More stable training with larger batch size
- Lower risk of overfitting
- Suitable for deployment scenarios

### **Analysis Points:**

1. **Does deeper backbone improve accuracy?**
   - Compare V2 vs V1 mAP scores
   - Check if additional layers help with complex damage patterns

2. **Can lighter model maintain reasonable performance?**
   - Compare V3 vs V1 trade-offs
   - Evaluate speed vs accuracy balance

3. **Overfitting analysis:**
   - Monitor train/val loss curves
   - Check early stopping triggers
   - Analyze gap between training and validation metrics

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
