# Assignment 2 Consultation Summary

## Core Principle (Most Important)

> “It is never about the accuracy. It is about the methodology.”

The lecturer repeatedly emphasized:

- Accuracy is NOT the main focus
- Correct methodology is the most important thing
- Fair and consistent experiments matter more than high scores

---

# 1. Image Classification Requirements

## 1.1 Consistent Dataset Split Across Experiments

This is one of the biggest points discussed.

If you compare different models (e.g. SVM vs ANN):

- They MUST use the same dataset split
- They MUST use the same training/testing conditions

Example:

| Experiment | Train | Validation | Test |
|---|---|---|---|
| SVM | 70% | 15% | 15% |
| ANN | 70% | 15% | 15% |

---

### Wrong Methodology Example

❌ Incorrect:

| Experiment | Train | Validation | Test |
|---|---|---|---|
| SVM | 80% | 0% | 20% |
| ANN | 60% | 20% | 20% |

Reason:
- The comparison becomes unfair
- Different models receive different amounts of training data

---

## 1.2 Dataset Size

Lecturer mentioned:

- Around 5000 samples is acceptable
- 10 classes is acceptable

---

## 1.3 Must Plot Training Curves

The lecturer specifically said:

> “I don't want to see just the accuracy. I want to see the curves.”

You should include:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

---

## 1.4 Analyze Overfitting / Underfitting

### Overfitting

Signs:
- Validation loss increases
- Training accuracy much higher than validation accuracy

Possible solutions:
- Remove layers
- Add dropout
- Add regularization
- Data augmentation

---

### Underfitting

Signs:
- Both training and validation performance are poor

Possible solutions:
- Add layers
- Train longer
- Improve architecture

---

## 1.5 Early Stopping is Allowed

Example:

```python
epochs = 100
early_stopping = True
```

If training stops at epoch 30:

✅ This is acceptable

As long as:
- Early stopping is justified
- Curves show overfitting prevention

---

# 2. Transfer Learning / CNN Requirements (VERY IMPORTANT)

This is one of the most critical marking points.

---

## 2.1 Do NOT Freeze Layers

Lecturer repeatedly said:

> “If you freeze it, zero.”

and

> “Wrong experiment.”

---

### ❌ Wrong Example

```python
for param in model.parameters():
    param.requires_grad = False
```

---

### ✅ Correct Approach

You MAY:
- Use pretrained weights
- Use transfer learning

BUT:

- ALL layers must be trainable
- ALL layers must be unfrozen

---

## 2.2 Correct Transfer Learning Method

Allowed:
- Initialize using pretrained weights
- Retrain ALL layers

Not allowed:
- Freeze backbone layers
- Train only final classifier

---

# 3. Custom Model Requirements

This is another important marking criterion.

---

## 3.1 Customization Means Modifying CNN Structure

Lecturer said:

> “Customization refers to deletion and addition of convolution layers.”

---

### ❌ NOT sufficient

Only changing:
- Dropout
- Learning rate
- Optimizer

does NOT count as customization.

---

### ✅ Correct Customization Examples

You should:
- Add convolution layers
- Remove convolution layers
- Modify CNN blocks

---

# 4. Object Detection Requirements (YOLO)

---

## 4.1 No Architecture Comparison Required

Lecturer clearly said:

> “No comparison required.”

Therefore:

You do NOT need:
- YOLOv5 vs YOLOv8
- FasterRCNN vs YOLO

---

## 4.2 More Epochs Are Normal

Examples mentioned:
- 200 epochs
- 300 epochs

are acceptable.

---

## 4.3 80% mAP is Acceptable

Lecturer appeared satisfied with:
- ~80% mAP performance

---

## 4.4 Confusion Matrix Purpose

Confusion matrix helps identify:
- Similar classes
- Misclassified categories

Example:
- Chair vs Sofa
- Dog vs Cat

---

## 4.5 Increase Resolution if Matrix Is Hard to Read

Suggested:
- Increase image resolution
- Make confusion matrix easier to analyze

---

## 4.6 Do NOT Resplit Detection Dataset

Lecturer said:

> “The split is already provided to you.”

If the dataset already contains:

- train/
- valid/
- test/

Then:

❌ Do NOT create your own split

---

# 5. Important Report Requirements

Your report should include:

- Dataset explanation
- Experiment methodology
- Training curves
- Evaluation metrics
- Confusion matrix
- Overfitting/underfitting discussion
- Architecture visualization

---

# 6. Common Mistakes That Cause Heavy Mark Deductions

## ❌ Freezing Layers

Most dangerous mistake.

Possible outcome:
- Zero marks for experiment

---

## ❌ Different Splits Across Experiments

Creates unfair comparison.

---

## ❌ Only Changing Dropout

Does NOT count as custom architecture.

---

## ❌ Submitting to Wrong Location

Example:
- Report not uploaded to Turnitin

Lecturer mentioned:
- Technically could receive zero

---

## ❌ Ignoring Specification Requirements

The lecturer strongly emphasized:

> “Follow instructions.”

---

# 7. What the Lecturer Actually Cares About

The consultation strongly suggests the lecturer values:

## Most Important

- Correct methodology
- Fair comparison
- Consistent data split
- Understanding ML concepts
- Following instructions

---

## Less Important

- Extremely high accuracy
- Minor performance differences

---

# 8. Final Checklist Before Submission

## ✅ Image Classification

- Same train/val/test split across experiments
- Curves included
- Overfitting analysis included

---

## ✅ Transfer Learning

- No frozen layers
- All layers trainable

---

## ✅ Custom Model

- CNN structure modified
- Conv layers added/removed

---

## ✅ Object Detection

- Use provided dataset split
- Include confusion matrix analysis
- Explain errors/misclassifications

---

# Final Takeaway

The biggest message from the consultation:

> Correct methodology is far more important than high accuracy.

The lecturer is willing to help students,
but experiments MUST follow the specification correctly.