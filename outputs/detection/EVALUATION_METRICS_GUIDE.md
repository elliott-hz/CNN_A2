# Detection Evaluation Metrics Guide

This document provides a comprehensive guide to understanding detection evaluation metrics used in the CNN_A3 project.

---

## 📊 Core Metrics Overview

### 1. mAP (Mean Average Precision)

**Definition**: The primary metric for object detection, measuring both classification accuracy and localization precision.

#### mAP@0.5
- **What it measures**: Average Precision at IoU threshold = 0.5
- **Interpretation**: 
  - > 0.90: Excellent detection quality
  - 0.75-0.90: Good detection quality
  - 0.50-0.75: Moderate detection quality
  - < 0.50: Poor detection quality, needs improvement
- **Use case**: Standard metric for most detection tasks

#### mAP@0.5:0.95
- **What it measures**: Average of AP values at IoU thresholds from 0.5 to 0.95 (step 0.05)
- **Formula**: `mean(AP@0.5, AP@0.55, AP@0.60, ..., AP@0.95)`
- **Interpretation**: More stringent than mAP@0.5, rewards precise localization
- **COCO Standard**: This is the official COCO challenge metric
- **Typical values**: Usually 10-20% lower than mAP@0.5

**Why both?**
- mAP@0.5: Easy to achieve, shows basic detection capability
- mAP@0.5:0.95: Hard to achieve, shows localization precision

---

### 2. Precision, Recall, F1-Score

#### Precision
- **Formula**: `TP / (TP + FP)`
- **Question answered**: "Of all detected objects, how many are correct?"
- **High precision**: Few false positives
- **Low precision**: Many false alarms

#### Recall
- **Formula**: `TP / (TP + FN)`
- **Question answered**: "Of all actual objects, how many did we detect?"
- **High recall**: Few missed detections
- **Low recall**: Many objects missed

#### F1-Score
- **Formula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Purpose**: Single metric balancing precision and recall
- **Best for**: Comparing models when both precision and recall matter

**Trade-off**:
- Increasing confidence threshold → Higher precision, lower recall
- Decreasing confidence threshold → Lower precision, higher recall

---

### 3. IoU (Intersection over Union)

**Formula**: `Area of Intersection / Area of Union`

```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
```

**Range**: 0 (no overlap) to 1 (perfect match)

**Thresholds**:
- IoU ≥ 0.5: Considered a correct detection (standard)
- IoU ≥ 0.75: High-quality localization
- IoU ≥ 0.90: Excellent localization

**IoU Statistics**:
- **Mean IoU**: Average localization quality across all matched detections
- **Median IoU**: Robust measure less affected by outliers
- **Std Dev**: Consistency of localization (lower = more consistent)
- **Distribution**: Shows full range of localization quality

---

## 🔍 How Metrics Are Calculated

### Step 1: Match Predictions to Ground Truths

For each image:
1. Sort predictions by confidence score (descending)
2. For each prediction:
   - Find ground truth with highest IoU (same class, same image)
   - If IoU ≥ threshold AND ground truth not yet matched:
     - Mark as True Positive (TP)
     - Mark ground truth as matched
   - Else:
     - Mark as False Positive (FP)
3. Unmatched ground truths = False Negatives (FN)

### Step 2: Calculate Precision-Recall Curve

For different confidence thresholds:
- Calculate cumulative TP, FP, FN
- Compute precision and recall at each threshold
- Plot Precision vs Recall curve

### Step 3: Calculate Average Precision (AP)

**11-Point Interpolation Method** (PASCAL VOC standard):
```python
AP = 0
for t in [0, 0.1, 0.2, ..., 1.0]:  # 11 recall levels
    max_precision = max(precision at recall >= t)
    AP += max_precision / 11
```

**Why interpolation?**
- Smooths out fluctuations in PR curve
- Standard method for fair comparison
- Less sensitive to small changes

### Step 4: Calculate mAP

- **Per-class AP**: Calculate AP for each class separately
- **mAP**: Average of AP across all classes
- **mAP@multiple IoUs**: Repeat for each IoU threshold, then average

---

## 📈 Interpreting Results

### Scenario 1: High mAP@0.5, Low mAP@0.5:0.95

**Example**: mAP@0.5 = 0.85, mAP@0.5:0.95 = 0.45

**Diagnosis**: 
- ✅ Model detects objects correctly (high recall)
- ❌ Bounding boxes are imprecise (low localization quality)

**Solutions**:
- Increase training epochs
- Use larger input resolution
- Try models with better localization (Faster R-CNN)
- Add bounding box regression augmentation

### Scenario 2: Low Precision, High Recall

**Example**: Precision = 0.60, Recall = 0.90

**Diagnosis**:
- ✅ Detects most objects
- ❌ Many false positives

**Solutions**:
- Increase confidence threshold
- Add harder negative mining
- Improve data quality (remove ambiguous labels)
- Use NMS with lower IoU threshold

### Scenario 3: High Precision, Low Recall

**Example**: Precision = 0.95, Recall = 0.50

**Diagnosis**:
- ✅ Detections are accurate
- ❌ Misses many objects

**Solutions**:
- Decrease confidence threshold
- Add more training data (especially rare cases)
- Use data augmentation
- Try larger model backbone

### Scenario 4: Class Imbalance

**Symptoms**: One class has much lower metrics than others

**Solutions**:
- Use class-weighted loss
- Oversample minority classes
- Collect more data for underrepresented classes
- Use focal loss

---

## 🎯 Benchmark Expectations

### Dog Face Detection (Single Class)

| Metric | Poor | Moderate | Good | Excellent |
|--------|------|----------|------|-----------|
| **mAP@0.5** | < 0.60 | 0.60-0.75 | 0.75-0.85 | > 0.85 |
| **mAP@0.5:0.95** | < 0.35 | 0.35-0.50 | 0.50-0.65 | > 0.65 |
| **Precision** | < 0.70 | 0.70-0.80 | 0.80-0.90 | > 0.90 |
| **Recall** | < 0.65 | 0.65-0.75 | 0.75-0.85 | > 0.85 |
| **Mean IoU** | < 0.50 | 0.50-0.60 | 0.60-0.70 | > 0.70 |

### Model Comparison Expectations

| Model | Expected mAP@0.5 | Expected mAP@0.5:0.95 | Strengths | Weaknesses |
|-------|------------------|-----------------------|-----------|------------|
| **YOLOv8** | 0.75-0.80 | 0.55-0.65 | Fast, balanced | May miss small objects |
| **Faster R-CNN** | 0.80-0.85 | 0.60-0.70 | Best accuracy, good localization | Slow inference |
| **SSD** | 0.72-0.78 | 0.50-0.60 | Good small object detection | Lower overall accuracy |

---

## 📊 Visualization Guide

### IoU Distribution Histogram

**What to look for**:
- **Right-skewed** (peak near 1.0): Excellent localization
- **Centered around 0.6-0.7**: Good localization
- **Left-skewed** (peak near 0.5): Poor localization, needs improvement
- **Bimodal**: Mix of easy and hard cases

**Action items**:
- Mean < 0.6: Improve bounding box regression
- High variance: Inconsistent performance, check data quality

### mAP vs IoU Threshold Curve

**What to look for**:
- **Gradual decline**: Consistent localization quality
- **Steep drop after 0.5**: Localization barely meets threshold
- **Flat curve**: Very precise localization (rare)

**Action items**:
- Steep decline: Focus on localization precision
- Flat but low: Improve detection capability first

### Per-Class Metrics Bar Chart

**What to look for**:
- **Balanced bars**: Consistent performance across classes
- **One class much lower**: Data imbalance or class-specific issues
- **Precision >> Recall**: Conservative detector
- **Recall >> Precision**: Aggressive detector

---

## 💡 Practical Tips

### Improving mAP@0.5

1. **Data Quality**:
   - Ensure accurate annotations
   - Remove ambiguous or incorrect labels
   - Balance class distribution

2. **Model Capacity**:
   - Use larger backbone if underfitting
   - Add more training epochs
   - Fine-tune learning rate

3. **Augmentation**:
   - Add diverse augmentations
   - Include scale variations
   - Use mosaic/mixup (YOLOv8)

### Improving mAP@0.5:0.95

1. **Localization Precision**:
   - Increase input resolution
   - Use models with FPN (Faster R-CNN, YOLOv8)
   - Add bounding box refinement

2. **Training Strategy**:
   - Train longer (more epochs)
   - Use lower learning rate in fine-tuning
   - Enable advanced augmentations

3. **Post-processing**:
   - Tune NMS IoU threshold
   - Adjust confidence threshold
   - Use soft-NMS or weighted box fusion

### Debugging Poor Performance

**Checklist**:
1. ✅ Verify data loading (visualize samples)
2. ✅ Check label format consistency
3. ✅ Monitor training/validation loss curves
4. ✅ Verify no data leakage between splits
5. ✅ Test with known-good images
6. ✅ Compare against baseline model

---

## 📚 Further Reading

- **COCO Evaluation**: https://cocodataset.org/#detection-eval
- **PASCAL VOC Metrics**: http://host.robots.ox.ac.uk/pascal/VOC/
- **mAP Explanation**: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
- **IoU and NMS**: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

---

**Last Updated**: 2026-04-26  
**Author**: CNN_A3 Project Team  
**Status**: ✅ Complete implementation
