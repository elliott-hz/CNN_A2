# Faster R-CNN DataLoader Implementation Guide

**Student ID:** 25509225  
**Date:** 2026-04-30  

---

## Overview

This document describes the implementation of the Faster R-CNN DataLoader for Assignment 2. The dataloader supports multiple annotation formats and provides a complete training pipeline.

---

## Architecture

### Components

1. **`FasterRCNNDataset`** - Custom PyTorch Dataset class
   - Loads images and annotations
   - Converts to Faster R-CNN format
   - Supports COCO, Pascal VOC, and YOLO formats

2. **`create_faster_rcnn_dataloaders()`** - Factory function
   - Creates train/val/test DataLoaders
   - Configures collate function for variable-size targets
   - Handles different annotation formats

3. **`exp02_detection_FasterRCNN.py`** - Experiment script
   - Orchestrates the full training pipeline
   - Uses the dataloader and trainer
   - Generates experiment summary

---

## Supported Annotation Formats

### 1. COCO Format

**Structure:**
```
data/25509225/Object_Detection/coco/
├── train/
│   ├── *.jpg (images)
│   └── train_annotations.json
├── valid/
│   ├── *.jpg (images)
│   └── valid_annotations.json
└── test/
    ├── *.jpg (images)
    └── test_annotations.json
```

**Annotation JSON Structure:**
```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "Cell"}
  ]
}
```

### 2. Pascal VOC Format

**Structure:**
```
data/25509225/Object_Detection/pascal/
├── train/
│   ├── *.jpg (images)
│   └── *.xml (annotations)
├── valid/
│   ├── *.jpg
│   └── *.xml
└── test/
    ├── *.jpg
    └── *.xml
```

**XML Annotation Example:**
```xml
<annotation>
  <filename>image1.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>Cell</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>200</xmax>
      <ymax>150</ymax>
    </bndbox>
  </object>
</annotation>
```

### 3. YOLO Format

**Structure:**
```
data/25509225/Object_Detection/yolo/
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**TXT Annotation Example:**
```
0 0.5 0.5 0.2 0.3  # class_id x_center y_center width height (normalized)
1 0.3 0.7 0.1 0.15
```

---

## Usage

### Basic Usage

```python
from src.data_processing.faster_rcnn_dataloader import create_faster_rcnn_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_faster_rcnn_dataloaders(
    data_root='data/25509225/Object_Detection/coco',
    batch_size=4,
    num_workers=2,
    annotation_format='coco',
    class_names=['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']
)

# Iterate over batches
for images, targets in train_loader:
    # images: List[Tensor(C, H, W)]
    # targets: List[Dict] with keys: boxes, labels, image_id, area, iscrowd
    pass
```

### Running the Experiment

```bash
# Run Faster R-CNN experiment
python experiments/exp02_detection_FasterRCNN.py
```

The experiment will:
1. Load dataset from `data/25509225/Object_Detection/coco/`
2. Initialize Faster R-CNN model with ResNet50+FPN backbone
3. Train for 50 epochs with early stopping
4. Save results to `outputs/exp02_fasterrcnn_TIMESTAMP/`

---

## Target Format

Faster R-CNN expects targets in this format:

```python
target = {
    'boxes': torch.FloatTensor(N, 4),     # [x1, y1, x2, y2] format
    'labels': torch.IntTensor(N),          # Class labels (1-indexed, 0=background)
    'image_id': torch.IntTensor([img_id]), # Image identifier
    'area': torch.FloatTensor(N),          # Box areas
    'iscrowd': torch.ByteTensor(N)         # Crowd flag (0 or 1)
}
```

**Important Notes:**
- Labels are 1-indexed (0 is reserved for background)
- Boxes are in `[x1, y1, x2, y2]` format (not center-width-height)
- All tensors must be on CPU (not GPU) before passing to model

---

## Configuration

### Experiment Parameters

```python
# In exp02_detection_FasterRCNN.py
STUDENT_ID = "25509225"
DATA_ROOT = f"data/{STUDENT_ID}/Object_Detection/coco"
ANNOTATION_FORMAT = 'coco'  # Options: 'coco', 'pascal', 'yolo'
CLASS_NAMES = ['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']

EPOCHS = 50
BATCH_SIZE = 4  # Smaller due to memory constraints
LR = 0.001
```

### Dataloader Parameters

```python
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,  # Custom collate for variable-size targets
    drop_last=True
)
```

---

## Custom Collate Function

Faster R-CNN requires a special collate function because each image can have a different number of objects:

```python
def collate_fn(batch):
    return tuple(zip(*batch))
```

This returns:
- `images`: List of tensors (not stacked)
- `targets`: List of dictionaries (not batched)

---

## Training Pipeline

The training loop in `FasterRCNNTrainer`:

```python
model.model.train()
for images, targets in train_loader:
    # Move to device
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # Forward pass (returns loss dict)
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    # Backward pass
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
```

**Loss Components:**
- `loss_classifier`: Classification loss
- `loss_box_reg`: Box regression loss
- `loss_objectness`: Objectness loss (RPN)
- `loss_rpn_box_reg`: RPN box regression loss

---

## Troubleshooting

### Issue 1: Dataset Not Found

**Error:**
```
FileNotFoundError: No JSON annotation files found in ...
```

**Solution:**
- Verify `DATA_ROOT` path is correct
- Check that annotation files exist in the expected format
- Ensure `annotation_format` matches your data

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `BATCH_SIZE` (try 2 or 1)
- Reduce image size in model config
- Use gradient accumulation
- Close other GPU applications

### Issue 3: Label Mismatch

**Error:**
```
IndexError: index out of range
```

**Solution:**
- Ensure `num_classes = len(CLASS_NAMES) + 1` (including background)
- Check that all class names in annotations match `CLASS_NAMES`
- Verify labels are 1-indexed (not 0-indexed)

### Issue 4: Empty Annotations

**Warning:**
```
Loaded 0 samples from ...
```

**Solution:**
- Check that image and annotation filenames match
- Verify annotation format is correct
- Ensure images exist in the specified directory

---

## Performance Tips

1. **Batch Size:**
   - Start with 4, reduce if OOM
   - Larger batches = faster training but more memory

2. **Num Workers:**
   - Set to 2-4 for CPU data loading
   - Too many workers can cause memory issues

3. **Image Size:**
   - Default 640x640 is good balance
   - Reduce to 512 for faster training
   - Increase to 800 for better accuracy

4. **Early Stopping:**
   - Patience=10 prevents overfitting
   - Adjust based on validation loss curve

---

## Comparison with YOLOv8

| Aspect | YOLOv8 | Faster R-CNN |
|--------|--------|--------------|
| **Dataloader** | Ultralytics built-in | Custom implementation |
| **Training** | Single command (`model.train()`) | Custom loop |
| **Speed** | Faster (single-stage) | Slower (two-stage) |
| **Accuracy** | Good | Potentially higher |
| **Memory** | Lower | Higher |
| **Complexity** | Low | High |

---

## Next Steps

1. **Run the experiment:**
   ```bash
   python experiments/exp02_detection_FasterRCNN.py
   ```

2. **Monitor training:**
   - Watch terminal output for loss values
   - Check `outputs/exp02_*/training/best_model.pth`

3. **Evaluate results:**
   - Review `experiment_summary.md`
   - Compare with YOLOv8 results (exp01)

4. **Improve evaluation:**
   - Implement proper mAP calculation
   - Add confusion matrix for detection
   - Visualize predictions on test set

---

## References

- [Torchvision Faster R-CNN Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)
- [YOLO Format](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data)

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks
