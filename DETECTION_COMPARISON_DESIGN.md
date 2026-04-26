# Detection Model Comparison Experiment Design

## 📋 Background and Rationale

### Current State Analysis

The project currently has **3 YOLOv8 variants** for dog face detection:
- `exp01_detection_YOLOv8_baseline.py` - Medium backbone (m), 640px input
- ~~`exp01_detection_YOLOv8_v1.py`~~ - **DELETED** (Large backbone, had bugs)
- ~~`exp01_detection_YOLOv8_v2.py`~~ - **DELETED** (Small backbone, had bugs)

**Problems Identified:**
1. ❌ **Limited Architecture Diversity**: All three experiments use the same YOLOv8 architecture family
2. ❌ **Code Quality Issues**: V1 had duplicate `output_dir` creation bug breaking resume functionality
3. ❌ **Incomplete Comparison**: Cannot compare fundamentally different detection paradigms (single-stage vs two-stage)
4. ❌ **Missing Classical Models**: No Faster R-CNN or SSD implementations for academic completeness

---

## 🎯 Proposed Solution

### New Experiment Structure

Replace the 3 YOLOv8 variants with **3 diverse detection architectures**:

| Experiment | Model | Architecture Type | Key Characteristics |
|------------|-------|-------------------|---------------------|
| **Exp01** | YOLOv8 (Medium) | Single-stage (Anchor-based) | Fast inference, balanced accuracy |
| **Exp02** | Faster R-CNN | Two-stage (Region Proposal) | Higher accuracy, slower inference |
| **Exp03** | SSD (VGG16) | Single-stage (Multi-scale) | Moderate speed, good small object detection |

### Why This Approach?

#### 1. **Academic Rigor** 📚

Comparing different detection paradigms provides:
- **Single-stage detectors** (YOLOv8, SSD): Direct bounding box prediction, faster but may miss small objects
- **Two-stage detectors** (Faster R-CNN): Region proposal + classification, higher accuracy but slower

This comparison is standard in computer vision research and provides meaningful insights into:
- Speed vs accuracy trade-offs
- Different feature extraction strategies
- Robustness to scale variations

#### 2. **Fair Comparison Requirements** ⚖️

To ensure valid comparisons, we must maintain:

**✅ Dataset Consistency:**
- Same train/val/test split across all models
- Only annotation format changes (YOLO → COCO/VOC)
- No data augmentation differences

**✅ Training Configuration Fairness:**
- All models trained ≥100 epochs (we use 150)
- Unified optimizer/scheduler framework where possible
- Model-specific hyperparameter tuning allowed (different architectures have different optimal settings)

**✅ Hardware Constraints:**
- All experiments run on T4 GPU (10GB VRAM)
- Batch size adjusted per model memory requirements
- Gradient accumulation used when needed

#### 3. **Technical Implementation Strategy** 🔧

**Why Separate Model/Trainer Files?**

Current architecture:
```
src/models/detection_model.py       → YOLOv8Detector (Ultralytics API)
src/training/detection_trainer.py   → DetectionTrainer (calls model.train())
```

**Problem**: Ultralytics YOLOv8 and Torchvision (Faster R-CNN/SSD) have **fundamentally different APIs**:

| Aspect | YOLOv8 (Ultralytics) | Faster R-CNN/SSD (Torchvision) |
|--------|----------------------|--------------------------------|
| **Training** | `model.train(data=...)` | Manual training loop required |
| **Loss Calculation** | Built-in automatic | Must implement manually |
| **Data Format** | YOLO (.txt files) | COCO JSON / VOC XML |
| **Augmentation** | Built-in (Mosaic, Mixup) | Custom transforms needed |
| **Evaluation** | `model.val()` | Custom evaluation code |

**Solution**: Keep them separate to avoid:
- ❌ Complex adapter patterns that add bugs
- ❌ Loss of framework-specific optimizations
- ❌ Unclear code responsibility

New structure:
```
src/models/
├── detection_model.py              # YOLOv8Detector (unchanged)
└── torchvision_detection.py        # NEW: FasterRCNNDetector, SSDDetector

src/training/
├── detection_trainer.py            # For YOLOv8 (unchanged)
└── torchvision_detection_trainer.py # NEW: Unified trainer for Faster R-CNN & SSD
```

#### 4. **Data Format Conversion** 🔄

**Current Format**: YOLO (.txt files)
```
data/processed/detection/
├── images/{train,val,test}/
└── labels/{train,val,test}/*.txt  # class x_center y_center width height
```

**Required Formats**:
- **Faster R-CNN**: COCO JSON format
- **SSD**: VOC XML format

**Conversion Script**: `src/data_processing/convert_detection_format.py`
- Reads YOLO format
- Outputs COCO JSON → `data/processed/detection_coco/`
- Outputs VOC XML → `data/processed/detection_voc/`
- **Guarantees**: Same train/val/test split, only structure changes

---

## 📊 Expected Benefits

### 1. **Comprehensive Model Comparison**

| Metric | YOLOv8 | Faster R-CNN | SSD |
|--------|--------|--------------|-----|
| **Inference Speed** | ⭐⭐⭐ Fastest | ⭐ Slowest | ⭐⭐ Moderate |
| **Accuracy (mAP)** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐ Good |
| **Small Objects** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐ Good |
| **Training Speed** | ⭐⭐⭐ Fastest | ⭐ Slowest | ⭐⭐ Moderate |
| **Memory Usage** | ⭐⭐ Moderate | ⭐ Highest | ⭐⭐ Low |

### 2. **Educational Value**

Students/researchers can learn:
- Different detection paradigm strengths/weaknesses
- How architecture affects performance
- Trade-offs in real-world deployment scenarios

### 3. **Research Completeness**

Covers the evolution of object detection:
- **2015**: Faster R-CNN (two-stage pioneer)
- **2016**: SSD (single-stage multi-scale)
- **2023**: YOLOv8 (modern single-stage)

---

## 🛠️ Implementation Plan

### Phase 1: Data Preparation
1. ✅ Delete old V1/V2 experiment files
2. ⏳ Create `convert_detection_format.py` script
3. ⏳ Run conversion to generate COCO and VOC datasets

### Phase 2: Model Implementation
4. ⏳ Create `torchvision_detection.py` with FasterRCNNDetector and SSDDetector
5. ⏳ Create `torchvision_detection_trainer.py` with unified training loop
6. ⏳ Implement proper loss calculation and gradient handling

### Phase 3: Experiment Scripts
7. ⏳ Create `exp02_detection_Faster-R-CNN.py`
8. ⏳ Create `exp03_detection_SSD.py`
9. ⏳ Configure model-specific hyperparameters

### Phase 4: Documentation
10. ⏳ Update `DETECTION_TRAINING.md` with new experiments
11. ⏳ Update `README.md` experiment overview
12. ⏳ Add comparison results table

---

## ⚠️ Important Considerations

### Memory Management (T4 10GB)

**Faster R-CNN Challenges:**
- Large backbone (ResNet50) + FPN = high memory usage
- Solution: `batch_size=4` + `gradient_accumulation_steps=4` → effective batch=16

**SSD Advantages:**
- Lighter backbone (VGG16) without FPN
- Can use larger batch: `batch_size=16`

**YOLOv8 Optimization:**
- Most memory-efficient architecture
- Can use: `batch_size=24`

### Training Time Estimates (T4 GPU)

| Model | Epochs | Estimated Time | Notes |
|-------|--------|----------------|-------|
| YOLOv8 | 120 | ~2-3 hours | Optimized implementation |
| Faster R-CNN | 150 | ~4-5 hours | Slower convergence |
| SSD | 150 | ~3-4 hours | Moderate speed |

### Hyperparameter Tuning Philosophy

**Unified Framework:**
- All use SGD with momentum (except YOLOv8 which uses Adam)
- All use cosine annealing or step decay schedulers
- All enable AMP (mixed precision)

**Model-Specific Adjustments:**
- Learning rates tuned per architecture (SGD typically needs higher LR than Adam)
- Weight decay adjusted based on model complexity
- Early stopping patience proportional to training duration

---

## 📈 Success Criteria

### Quantitative Metrics
- All models achieve mAP@0.5 > 0.70
- Training completes without OOM errors on T4
- Reproducible results with fixed random seeds

### Qualitative Goals
- Clear documentation of each model's characteristics
- Meaningful performance comparison insights
- Clean, maintainable code architecture

---

## 🔗 Related Documentation

- [DETECTION_TRAINING.md](DETECTION_TRAINING.md) - Detailed training configurations
- [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) - Dataset preparation workflow
- [README.md](README.md) - Project overview and quick start

---

**Document Version**: 1.0  
**Created**: 2026-04-26  
**Author**: CNN_A3 Project Team  
**Status**: 📝 Planning Phase
