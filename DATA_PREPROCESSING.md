# Data Preprocessing Guide

This document provides detailed information about dataset sources, preprocessing workflows, and data format specifications for the Visual Dog Emotion Recognition system.

---

## 📊 Dataset Sources

### Detection Dataset (Dog Face Detection)

**Source**: Kaggle - Dog Face Detection Dataset

**Raw Structure**:
```
data/raw/detection_dataset/
├── train_img/           # ~5,924 training images (.jpg)
├── train_label/         # Training labels (YOLO format .txt)
├── val_img/             # ~230 validation images (.jpg)
└── val_label/           # Validation labels (YOLO format .txt)
```

**Label Format** (YOLO):
```
class_id x_center y_center width height
```
- All coordinates normalized to [0, 1]
- Each `.txt` file may contain multiple bounding boxes (multi-dog images)
- Single class: dog (class_id = 0)

**Example** (`img_00000.txt`):
```
0 0.523456 0.412345 0.156789 0.234567
0 0.234567 0.678901 0.098765 0.123456
```

### Emotion Dataset (Dog Emotion Classification)

**Source**: Kaggle - Dog Emotions Dataset (5 classes)

**Raw Structure**:
```
data/raw/emotion_dataset/
├── alert/      # ~1,865 images (alert emotion)
├── angry/      # ~1,865 images (angry emotion)
├── frown/      # ~1,865 images (frown emotion)
├── happy/      # ~1,865 images (happy emotion)
└── relax/      # ~1,865 images (relaxed emotion)
```

**Classes**: 5 emotion categories
- Total: ~9,325 images
- Balanced distribution across classes

---

## 🔄 Preprocessing Workflow

### Overview

The project uses a **minimal data preparation workflow**:
1. **Manual Download**: You must download and extract datasets to `data/raw/` before running any experiments
2. **Lightweight Parsing**: Run the parsing script to organize data paths and create train/val/test splits
3. **No Image Preprocessing**: Images are NOT resized, normalized, or augmented during preprocessing - they're loaded on-the-fly during training with augmentations applied

**Key Benefits:**
- ✅ No memory issues (images not loaded into RAM)
- ✅ Fast setup (seconds instead of minutes)
- ✅ Flexible (preprocessing happens in training pipeline with augmentations)

### Running the Preprocessing Script

```bash
bash scripts/run_data_preprocessing.sh
```

This script performs two main tasks:

#### Task 1: Parse Detection Dataset

**Script**: [`src/data_processing/detection_preprocessor.py`](src/data_processing/detection_preprocessor.py)

**What it does:**
1. Loads training and validation images from `train_img/` and `val_img/`
2. Reads YOLO-format annotations from corresponding label directories
3. Combines all images and creates stratified train/val/test splits (70/20/10)
4. Saves lightweight JSON metadata to `data/splitting/detection_split/`

**Output Structure**:
```
data/splitting/detection_split/
├── train_split.json    # Training split metadata (image paths + labels)
├── val_split.json      # Validation split metadata
├── test_split.json     # Test split metadata
└── metadata.json       # Overall dataset statistics
```

**JSON Format Example** (`train_split.json`):
```json
[
  {
    "image_path": "data/raw/detection_dataset/train_img/img_00000.jpg",
    "label_path": "data/raw/detection_dataset/train_label/img_00000.txt"
  },
  ...
]
```

#### Task 2: Parse Emotion Dataset

**Script**: [`src/data_processing/emotion_preprocessor.py`](src/data_processing/emotion_preprocessor.py)

**What it does:**
1. Scans all subdirectories in `data/raw/emotion_dataset/`
2. Organizes image paths by class label
3. Creates stratified train/val/test splits (70/20/10) maintaining class balance
4. Saves lightweight JSON metadata to `data/splitting/emotion_split/`

**Output Structure**:
```
data/splitting/emotion_split/
├── train_split.json    # Training split metadata (image paths + labels)
├── val_split.json      # Validation split metadata
├── test_split.json     # Test split metadata
└── metadata.json       # Class distribution and statistics
```

**JSON Format Example** (`train_split.json`):
```json
[
  {
    "image_path": "data/raw/emotion_dataset/happy/img_001.jpg",
    "label": "happy"
  },
  ...
]
```

### Expected Output

After running the preprocessing script, you should see output like:

```
==========================================
Running Data Preprocessing
==========================================

[1/2] Parsing Detection Dataset...
================================================================================
DETECTION DATASET PARSING AND SPLITTING
================================================================================

[1/4] Loading training data...
  Loaded 5924 training images

[2/4] Loading validation data...
  Loaded 230 validation images

[3/4] Combining and splitting dataset (70/20/10)...

[4/4] Saving split metadata...
  Saved train split: 4307 images
  Saved val split: 1231 images
  Saved test split: 616 images
  Saved metadata

PARSING AND SPLITTING COMPLETE
Total samples: 6154
  Train: 4307 images
  Valid: 1231 images
  Test: 616 images

Note: Images are NOT preprocessed. They will be loaded during training.

[2/2] Parsing Emotion Dataset...
================================================================================
EMOTION DATASET PARSING AND SPLITTING
================================================================================

[1/3] Loading and organizing dataset...
  Loaded 9325 images across 5 classes

  Class distribution:
    alert: 1865
    angry: 1865
    frown: 1865
    happy: 1865
    relax: 1865

[2/3] Splitting dataset (70/20/10)...

[3/3] Saving split metadata...
  Saved train split: 6527 images
  Saved val split: 1865 images
  Saved test split: 933 images
  Saved metadata

PARSING AND SPLITTING COMPLETE
Total samples: 9325
  Train: 6527 images
  Valid: 1865 images
  Test: 933 images

Note: Images are NOT preprocessed. They will be loaded during training.

==========================================
Data parsing complete!
==========================================
```

---

## 📁 Processed Data Structure

After preprocessing, your data directory looks like this:

```
data/
├── raw/                           # Original downloaded datasets (unchanged)
│   ├── detection_dataset/
│   │   ├── train_img/
│   │   ├── train_label/
│   │   ├── val_img/
│   │   └── val_label/
│   └── emotion_dataset/
│       ├── alert/
│       ├── angry/
│       ├── frown/
│       ├── happy/
│       └── relax/
│
└── splitting/                     # Lightweight JSON metadata (no image copying)
    ├── detection_split/
    │   ├── train_split.json       # Paths to 4,307 training images
    │   ├── val_split.json         # Paths to 1,231 validation images
    │   ├── test_split.json        # Paths to 616 test images
    │   └── metadata.json          # Dataset statistics
    └── emotion_split/
        ├── train_split.json       # Paths to 6,527 training images
        ├── val_split.json         # Paths to 1,865 validation images
        ├── test_split.json        # Paths to 933 test images
        └── metadata.json          # Class distribution stats
```

**Important Notes:**
- ❌ **No image files are copied or moved** - only JSON metadata is created
- ✅ **Original images remain in `data/raw/`** - saves disk space
- ✅ **Fast preprocessing** - completes in seconds, not minutes
- ✅ **Flexible loading** - images loaded on-the-fly during training with augmentations

---

## 🔧 Creating Small Subsets for Testing

For quick testing and debugging, you can create small subsets:

```bash
bash scripts/run_data_preprocessing.sh --create-subset
```

Customize the number of samples:
```bash
bash scripts/run_data_preprocessing.sh \
    --create-subset \
    --train-samples 50 \
    --val-samples 10 \
    --test-samples 10
```

This creates a minimal dataset in `data/processed/detection_small/` for rapid iteration.

---

## 📋 Data Format Specifications

### Image Specifications

**Detection Dataset Images:**
- **Format**: JPEG (quality ~95, default OpenCV)
- **Size**: Original dimensions preserved (NOT resized during preprocessing)
- **Color Space**: BGR (OpenCV default, converted to RGB during training)
- **Naming**: Sequential `img_XXXXX.jpg`

**Emotion Dataset Images:**
- **Format**: JPEG/PNG (original format preserved)
- **Size**: Original dimensions preserved (resized to 224×224 during training)
- **Color Space**: RGB (converted during training if needed)

### Annotation Specifications

**Detection Labels (YOLO Format):**
- **File Extension**: `.txt`
- **Encoding**: UTF-8
- **Structure**: One line per bounding box
- **Coordinate System**: Normalized [0, 1] relative to image dimensions
- **Format**: `class_id x_center y_center width height`

**Emotion Labels:**
- Stored as directory names (e.g., `happy/`, `angry/`)
- Mapped to integer indices during training:
  - `alert` → 0
  - `angry` → 1
  - `frown` → 2
  - `happy` → 3
  - `relax` → 4

---

## ⚠️ Important Notes

### What Preprocessing Does NOT Do

1. **❌ No Image Resizing**: Images retain original dimensions
2. **❌ No Normalization**: Pixel values not scaled to [0, 1]
3. **❌ No Augmentation**: No rotations, flips, or color jittering
4. **❌ No Image Copying**: Only JSON metadata created, no file duplication

### What Happens During Training

All image transformations happen **on-the-fly** during training:

**Detection Training (YOLOv8):**
- Automatic letterbox resize to preserve aspect ratio
- Built-in augmentations (mosaic, mixup, etc.)
- Normalization to [0, 1]

**Classification Training:**
- Resize to 224×224 (or model-specific size)
- Random horizontal flip, rotation, color jitter
- Normalization using ImageNet mean/std
- Center crop for validation/testing

This approach ensures:
- ✅ **Memory efficiency**: Only one image in RAM at a time
- ✅ **Flexibility**: Easy to change augmentations without re-preprocessing
- ✅ **Reproducibility**: Same raw images, different augmentations per epoch

---

## 🔍 Troubleshooting

### Common Issues

**Problem**: "Dataset not found" error
```
Solution: Verify datasets exist in data/raw/ with correct structure
- data/raw/detection_dataset/train_img/*.jpg
- data/raw/emotion_dataset/happy/*.jpg
```

**Problem**: Empty JSON files after preprocessing
```
Solution: Check that raw data directories contain images
ls data/raw/detection_dataset/train_img/ | wc -l
ls data/raw/emotion_dataset/happy/ | wc -l
```

**Problem**: Class imbalance warnings
```
Solution: The preprocessing script uses stratified splitting to maintain balance
Check metadata.json for actual distribution
```

### Verifying Preprocessed Data

```python
import json

# Check detection split
with open('data/splitting/detection_split/train_split.json') as f:
    train_data = json.load(f)
    print(f"Training samples: {len(train_data)}")
    print(f"Sample entry: {train_data[0]}")

# Check emotion split
with open('data/splitting/emotion_split/metadata.json') as f:
    metadata = json.load(f)
    print(f"Class distribution: {metadata['class_distribution']}")
```

---

## 📚 Related Documentation

- **Model Training**: See [MODEL_TRAINING.md](MODEL_TRAINING.md) for experiment configurations
- **Web Application**: See [MODEL_APPLICATION.md](MODEL_APPLICATION.md) for inference pipeline
- **Project Overview**: See [README.md](README.md) for architecture summary
