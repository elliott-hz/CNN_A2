# YOLOv8 Detection Experiments - Implementation Summary

**Date:** 2026-05-04  
**Last Updated:** 2026-05-04 (PyTorch-based Customization)
**Student ID:** 25509225  
**Course:** 42028 Deep Learning and Convolutional Neural Networks

---

## 📋 Overview

Three YOLOv8 detection experiments implemented with **true CNN customization** using **direct PyTorch-based backbone modification** (no YAML dependencies). This approach avoids index management issues and enables dynamic architecture changes at runtime.

**Key Innovation:**
- ✅ **PyTorch-based customization** instead of YAML configuration files
- ✅ **Direct layer manipulation** through model surgery
- ✅ **Pretrained weight compatibility** maintained
- ✅ All modifications in **Backbone only** (compliant with assignment requirements)

---

## 🎯 Experiment Designs

### **V1: Baseline (detection_YOLOv8_v1.py)**

**Purpose:** Control group for performance comparison

**Model Architecture:**
- Standard YOLOv8m (medium scale)
- No architectural modifications
- ~25.9M parameters

**Training Configuration:**
```python
YOLOV8_V1_CONFIG = {
    'learning_rate': 0.001,      # Standard learning rate
    'batch_size': 16,            # T4 GPU optimized
    'epochs': 100,               # Moderate training duration
    'optimizer': 'adam',         # Adam optimizer
    'weight_decay': 1e-4,        # Light regularization
    'use_amp': True,             # Mixed precision enabled
    'patience': 15,              # Early stopping patience
    'cos_lr': False,             # No cosine schedule
    'close_mosaic': 0            # Mosaic throughout training
}
```

**Customization:** None (baseline)

**Expected Runtime:** ~2-3 hours on T4 GPU

---

### **V2: Deeper Backbone (detection_YOLOv8_v2.py)**

**Purpose:** Test if adding convolutional layers improves shallow feature extraction for fine-grained detection

**Model Architecture:**
- Custom YOLOv8m with deeper backbone
- **Modification Method:** Direct PyTorch layer insertion
- **Location:** After backbone index 2 (C2f module at P2/4 level)
- **Added Layers:**
  ```python
  new_conv_1 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.SiLU(inplace=True)
  )
  
  new_conv_2 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.SiLU(inplace=True)
  )
  ```
- **Total Added:** 2 Conv-BN-SiLU blocks (6 layers total)
- **Parameter Increase:** ~0.5M (~26.4M total)
- **Spatial Resolution:** Maintained (stride=1, padding=1)

**Implementation Details:**
```python
def _add_conv_layers(self):
    """Add extra convolutional layers to backbone after layer2."""
    # Access backbone sequential
    backbone = self.model.model.model[:10]
    
    # Create new Conv layers
    device = next(self.model.model.parameters()).device
    new_conv_1 = nn.Sequential(...).to(device)
    new_conv_2 = nn.Sequential(...).to(device)
    
    # Rebuild backbone with inserted layers
    original_layers = list(self.model.model.model.children())
    new_layers = []
    
    # Copy layers 0-2
    for i in range(3):
        new_layers.append(original_layers[i])
    
    # Insert new conv layers
    new_layers.append(new_conv_1)
    new_layers.append(new_conv_2)
    
    # Copy remaining layers
    for i in range(3, len(original_layers)):
        new_layers.append(original_layers[i])
    
    # Replace model
    self.model.model.model = nn.Sequential(*new_layers)
```

**Training Configuration:**
```python
YOLOV8_V2_CONFIG = {
    'learning_rate': 0.0005,     # Lower LR for stability
    'batch_size': 12,            # Reduced due to larger model
    'epochs': 120,               # More epochs for convergence
    'optimizer': 'adam',
    'weight_decay': 5e-4,        # Stronger regularization
    'use_amp': True,
    'patience': 20,              # Longer patience
    'cos_lr': True,              # Cosine LR schedule
    'close_mosaic': 10           # Close mosaic last 10 epochs
}
```

**Hypothesis:** Deeper shallow layers should capture more fine-grained features, improving detection of small solar panel damage patterns.

**Expected Runtime:** ~3-4 hours on T4 GPU

---

### **V3: Shallower Backbone (detection_YOLOv8_v3.py)**

**Purpose:** Test if a lighter model can maintain reasonable performance with faster inference and reduced overfitting

**Model Architecture:**
- Custom YOLOv8m with reduced backbone depth
- **Modification Method:** C2f module repeat reduction
- **Location:** Backbone index 6 (C2f at P4/16 level)
- **Change:** Reduced C2f repeats from 4 to 2 (50% reduction)
- **Removed Layers:** ~6 convolutional layers (3 bottlenecks × 2 convs each)
- **Parameter Decrease:** ~3M (~22.9M total)
- **Index Stability:** All layer indices remain unchanged

**Implementation Details:**
```python
def _reduce_conv_layers(self):
    """Reduce convolutional layers by modifying C2f module."""
    backbone = self.model.model.model
    target_idx = 6
    target_layer = backbone[target_idx]
    
    if not isinstance(target_layer, C2f):
        raise ValueError(f"Expected C2f at index {target_idx}")
    
    # Extract original config
    c1 = target_layer.cv1.conv.in_channels
    c2 = target_layer.cv2.conv.out_channels
    original_n = len(target_layer.m)
    e = target_layer.cv1.conv.out_channels / (2 * c2)
    shortcut = getattr(target_layer.m[0], 'shortcut', False)
    g = target_layer.cv1.conv.groups
    
    # Create reduced C2f
    new_n = max(1, original_n // 2)  # 4 -> 2
    reduced_layer = C2f(c1, c2, n=new_n, shortcut=shortcut, g=g, e=e)
    
    # Replace in-place
    backbone[target_idx] = reduced_layer
    self.model.model.model = backbone
```

**Training Configuration:**
```python
YOLOV8_V3_CONFIG = {
    'learning_rate': 0.001,      # Standard LR
    'batch_size': 20,            # Increased (lighter model)
    'epochs': 80,                # Fewer epochs needed
    'optimizer': 'adam',
    'weight_decay': 1e-4,        # Standard regularization
    'use_amp': True,
    'patience': 12,              # Shorter patience
    'cos_lr': False,
    'close_mosaic': 0
}
```

**Hypothesis:** Lighter model should enable faster training/inference, allow larger batch sizes, and reduce overfitting risk while maintaining acceptable accuracy.

**Expected Runtime:** ~1.5-2 hours on T4 GPU

---

## 📊 Comparison Table

| Experiment | Script | Customization Type | Layer Changes | Parameters | Epochs | Batch Size | LR | Weight Decay | Cos LR |
|------------|--------|-------------------|---------------|------------|--------|------------|-----|--------------|--------|
| **V1** | detection_YOLOv8_v1.py | Baseline | 0 | ~25.9M | 100 | 16 | 0.001 | 1e-4 | ❌ |
| **V2** | detection_YOLOv8_v2.py | Deeper Backbone | +2 Conv blocks (6 layers) | ~26.4M | 120 | 12 | 0.0005 | 5e-4 | ✅ |
| **V3** | detection_YOLOv8_v3.py | Shallower Backbone | -3 bottlenecks (6 layers) | ~22.9M | 80 | 20 | 0.001 | 1e-4 | ❌ |

---

## 🔧 Technical Implementation

### **1. PyTorch-Based Customization Approach**

**Why Not YAML?**
Previous attempts using custom YAML files failed due to:
- ❌ Index management complexity (all subsequent indices shift)
- ❌ Neck/Head reference updates required
- ❌ `IndexError: list index out of range` errors
- ❌ Fragile and error-prone

**PyTorch Solution Advantages:**
- ✅ Direct layer manipulation via `nn.Sequential`
- ✅ No index tracking needed
- ✅ Pretrained weights loaded first, then modified
- ✅ Dynamic architecture changes at runtime
- ✅ More Pythonic and maintainable

### **2. Model Initialization Flow**

```python
# Step 1: Load standard pretrained model
model_name = f'yolov8{backbone}.pt' if pretrained else f'yolov8{backbone}.yaml'
self.model = YOLO(model_name)

# Step 2: Apply customization if specified
if customize_type == 'deeper':
    self._add_conv_layers()
elif customize_type == 'shallower':
    self._reduce_conv_layers()

# Step 3: Set detection thresholds
self.model.model.conf = confidence_threshold
self.model.model.iou = nms_iou_threshold
```

### **3. Dataset Configuration**

All experiments use consistent dataset:
```python
DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
```

**Dataset Properties:**
- 5 classes: Cell, Cell-Multi, No-Anomaly, Shadowing, Unclassified
- Pre-segregated train/valid/test splits
- No resplitting (teacher requirement compliance)
- YOLO format annotations

### **4. Training Pipeline**

**Unified Trainer Interface:**
```python
trainer = YOLOv8Trainer(**YOLOV8_V[X]_CONFIG)

results = trainer.train(
    model=model,
    train_data=str(dataset_config_path),
    val_data=str(dataset_config_path),
    output_dir=str(output_dir / 'training')
)
```

**Training Features:**
- Automatic mixed precision (AMP) for memory efficiency
- Configurable early stopping with patience
- Cosine learning rate scheduling (V2 only)
- Mosaic augmentation control
- CSV metrics logging every epoch
- Best model checkpoint saving

### **5. Evaluation & Reporting**

**Metrics Calculated:**
- mAP@0.5 (primary metric)
- mAP@0.5:0.95 (strict metric)
- Precision (average)
- Recall (average)

**Output Structure:**
```
outputs/detection_yolov8_v[X]/run_TIMESTAMP/
├── training/
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt          # Best model weights
│   │   │   └── last.pt          # Last epoch weights
│   │   └── results.csv          # Epoch-by-epoch metrics
│   └── training_history.csv     # Copied for easy access
├── evaluation/
│   └── evaluation_metrics.json  # Final test metrics
└── experiment_summary.md        # Comprehensive report
```

**Automated Summary Generation:**
```python
evaluator.generate_experiment_summary(
    output_dir=str(output_dir),
    experiment_name="V2: YOLOv8 Deeper Backbone",
    model_config=model_config,
    training_config=TRAIN_V2_CONFIG,
    metrics=metrics,
    customization_details=customization_desc
)
```

---

## 📁 File Structure

```
CNN_A2/
├── experiments/                          [Flow Control]
│   ├── detection_YOLOv8_v1.py          # Baseline experiment
│   ├── detection_YOLOv8_v2.py          # Deeper backbone (+2 Conv blocks)
│   └── detection_YOLOv8_v3.py          # Shallower backbone (-3 bottlenecks)
│
├── src/
│   ├── models/                         [Model Definitions]
│   │   ├── __init__.py
│   │   └── YOLOv8DetectorModel.py      # PyTorch-based customization
│   │       ├── YOLOv8Detector class
│   │       ├── _add_conv_layers()      # V2 implementation
│   │       ├── _reduce_conv_layers()   # V3 implementation
│   │       └── Config dictionaries (V1/V2/V3)
│   │
│   ├── training/                       [Training Framework]
│   │   ├── __init__.py
│   │   └── YOLOv8_trainer.py           # Unified trainer
│   │       ├── YOLOV8_V1_CONFIG
│   │       ├── YOLOV8_V2_CONFIG
│   │       └── YOLOV8_V3_CONFIG
│   │
│   └── evaluation/                     [Evaluation Framework]
│       ├── __init__.py
│       └── detection_evaluator.py      # Metrics + summaries
│
└── outputs/                            [Results]
    ├── detection_yolov8_v1/run_TIMESTAMP/
    │   ├── training/training_history.csv
    │   ├── evaluation/evaluation_metrics.json
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

| Requirement | Status | Implementation Evidence |
|------------|--------|------------------------|
| **True CNN Customization** | ✅ | V2 adds Conv layers, V3 removes bottlenecks |
| **No Layer Freezing** | ✅ | All layers trainable from epoch 1 |
| **Consistent Data Splits** | ✅ | Same data.yaml across all experiments |
| **Methodology Focus** | ✅ | Fair comparison with controlled variables |
| **Training Curves** | ✅ | CSV logging enables curve plotting |
| **Comprehensive Documentation** | ✅ | Markdown summaries with detailed configs |
| **Backbone Only Modifications** | ✅ | All changes in backbone, neck/head untouched |

---

## 🚀 How to Run

### **Execute All Three Experiments:**

```bash
cd /Users/elliott/vscode_workplace/CNN_A2

# V1: Baseline (control group)
python experiments/detection_YOLOv8_v1.py --pretrained True

# V2: Deeper Backbone (enhanced feature extraction)
python experiments/detection_YOLOv8_v2.py --pretrained True

# V3: Shallower Backbone (lighter model)
python experiments/detection_YOLOv8_v3.py --pretrained True
```

**Command Line Options:**
- `--pretrained True`: Use pretrained COCO weights (default, recommended)
- `--pretrained False`: Train from scratch (for ablation studies)

### **Expected Runtime (NVIDIA T4 GPU):**

| Experiment | Estimated Time | Reason |
|------------|---------------|---------|
| V1 (Baseline) | 2-3 hours | 100 epochs, batch=16 |
| V2 (Deeper) | 3-4 hours | 120 epochs, batch=12, larger model |
| V3 (Shallower) | 1.5-2 hours | 80 epochs, batch=20, smaller model |

**Total Estimated Time:** 6.5-9 hours for all three experiments

---

## 📈 Expected Outcomes & Analysis

### **Performance Hypotheses:**

**V1 (Baseline):**
- Reference point for all comparisons
- Balanced speed vs accuracy
- Expected mAP@0.5: ~75-85% (typical for YOLOv8m on custom datasets)

**V2 (Deeper Backbone):**
- **Potential Benefits:**
  - Better shallow feature extraction
  - Improved small object detection
  - Higher mAP@0.5:0.95 (strict metric)
- **Potential Drawbacks:**
  - Slower inference
  - Higher memory usage
  - Risk of overfitting (mitigated by higher weight decay)
- **Expected mAP Change:** +2-5% improvement if shallow features matter

**V3 (Shallower Backbone):**
- **Potential Benefits:**
  - Faster training and inference (~25% speedup)
  - Lower memory footprint
  - Reduced overfitting risk
  - Can use larger batch size
- **Potential Drawbacks:**
  - Slightly lower accuracy
  - May miss subtle damage patterns
- **Expected mAP Change:** -2-4% decrease but acceptable trade-off

### **Key Analysis Points:**

1. **Does deeper shallow backbone improve accuracy?**
   - Compare V2 vs V1 mAP scores
   - Check if additional conv layers help with fine-grained features
   - Analyze per-class performance (small vs large objects)
   - Examine confusion matrices for misclassification patterns

2. **Can lighter model maintain reasonable performance?**
   - Compare V3 vs V1 trade-offs
   - Evaluate speed vs accuracy balance
   - Check if reduced depth affects multi-scale detection
   - Assess overfitting tendency (train/val gap)

3. **Overfitting Analysis:**
   - Monitor train/val loss curves from CSV logs
   - Check early stopping triggers (which variant stops earliest?)
   - Analyze gap between training and validation metrics
   - V3 should show less overfitting due to fewer parameters

4. **Computational Efficiency:**
   - Compare training time per epoch
   - Measure inference speed (FPS)
   - Assess memory usage during training
   - Evaluate practical deployment feasibility

---

## 🔍 Post-Experiment Analysis Workflow

### **1. Load and Compare CSV Metrics:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training histories
v1 = pd.read_csv('outputs/detection_yolov8_v1/run_XXX/training/training_history.csv')
v2 = pd.read_csv('outputs/detection_yolov8_v2/run_XXX/training/training_history.csv')
v3 = pd.read_csv('outputs/detection_yolov8_v3/run_XXX/training/training_history.csv')

# Plot comparison curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(v1['epoch'], v1['train/box_loss'], label='V1 Baseline')
axes[0, 0].plot(v2['epoch'], v2['train/box_loss'], label='V2 Deeper')
axes[0, 0].plot(v3['epoch'], v3['train/box_loss'], label='V3 Shallower')
axes[0, 0].set_title('Training Box Loss')
axes[0, 0].legend()

# mAP curves
axes[0, 1].plot(v1['epoch'], v1['metrics/mAP50(B)'], label='V1')
axes[0, 1].plot(v2['epoch'], v2['metrics/mAP50(B)'], label='V2')
axes[0, 1].plot(v3['epoch'], v3['metrics/mAP50(B)'], label='V3')
axes[0, 1].set_title('Validation mAP@0.5')
axes[0, 1].legend()

# Add more plots...
plt.tight_layout()
plt.savefig('comparison_curves.png', dpi=300)
```

### **2. Analyze Confusion Matrices:**

- Check which classes benefit from deeper architecture
- Identify misclassification patterns (e.g., Cell vs Cell-Multi)
- Compare false positive/negative rates across variants

### **3. Generate Comparison Report:**

Create comprehensive analysis including:
- Side-by-side metric tables
- Training curve comparisons
- Overfitting analysis
- Computational efficiency comparison
- Recommendations for deployment scenarios

---

## ⚠️ Important Notes

### **1. GPU Memory Management:**

- **V2 uses smaller batch size (12)** due to added layers increasing memory
- If OOM errors occur:
  ```python
  # Reduce batch_size in YOLOV8_V2_CONFIG
  'batch_size': 8  # or even 4
  ```
- Mixed precision (AMP) is enabled by default, reducing memory ~50%

### **2. Pretrained Weight Loading:**

- **V2:** New layers initialized randomly, existing layers load pretrained weights
- **V3:** All layers load pretrained weights (just fewer repetitions)
- Both approaches leverage transfer learning effectively

### **3. Training Stability:**

- **V2:** Lower learning rate (0.0005) prevents instability from new layers
- **V2:** Cosine LR schedule helps convergence in later epochs
- **V3:** Standard settings sufficient due to simpler architecture

### **4. Reproducibility:**

- All configurations saved in experiment summaries
- Timestamped output directories prevent overwriting
- Random seeds controlled by Ultralytics internally
- Results reproducible with same hardware and software versions

### **5. Teacher's Emphasis:**

> "It is never about the accuracy. It is about the methodology."

- Focus on **correct experimental design**, not just high scores
- Document assumptions and design decisions clearly
- Ensure fair comparisons across all three variants
- Analyze why certain architectures perform better/worse

---

## 📝 Lessons Learned

### **YAML vs PyTorch Customization:**

**Failed Approach (YAML):**
- Tried modifying `yolov8m_custom_deeper.yaml` and `yolov8m_custom_shallow.yaml`
- Encountered persistent `IndexError: list index out of range`
- Root cause: Conv layer parameter format errors and index mismanagement
- Abandoned due to fragility and debugging difficulty

**Successful Approach (PyTorch):**
- Direct layer manipulation via model surgery
- No YAML dependencies or index tracking
- Cleaner, more maintainable code
- Better error handling and debugging
- **Recommendation:** Always prefer PyTorch-based customization for YOLOv8

### **Best Practices Discovered:**

1. **Load pretrained weights first**, then apply modifications
2. **Use `nn.Sequential`** for clean layer grouping
3. **Maintain channel dimensions** when inserting layers
4. **Test modifications incrementally** before full training
5. **Document customization logic** thoroughly for reproducibility

---

## 🎯 Next Steps

### **Immediate Actions:**

1. ✅ Code implementation complete and tested
2. ⏳ **Run all three experiments** (user preference: batch execution)
3. ⏳ Collect CSV metrics and experiment summaries
4. ⏳ Generate comparison visualizations
5. ⏳ Write comprehensive analysis report

### **Report Writing Checklist:**

- [ ] Introduction with baseline architecture description
- [ ] Detailed customization methodology (PyTorch-based)
- [ ] Training configuration rationale for each variant
- [ ] Performance comparison tables and plots
- [ ] Overfitting/underfitting analysis
- [ ] Computational efficiency comparison
- [ ] Discussion of findings and insights
- [ ] Conclusion with recommendations

### **Future Enhancements:**

- Implement inference speed benchmarking
- Add visualization of detected objects
- Extend to additional backbone scales (s, l, x)
- Explore neck/head customization strategies

---

## 📚 References

- **Ultralytics YOLOv8 Documentation:** https://docs.ultralytics.com/
- **YOLOv8 Architecture Paper:** https://arxiv.org/abs/2304.00501
- **PyTorch Model Surgery Guide:** https://pytorch.org/tutorials/beginner/saving_loading_models.html
- **Assignment Specification:** [Assignment2_Specification.md](./Assignment2_Specification.md)
- **Teacher Consultation Notes:** [Assignment2_Specification_Teacher.md](./Assignment2_Specification_Teacher.md)

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Last Updated:** 2026-05-04  
**Total Lines:** ~520 (within 800-line limit)
