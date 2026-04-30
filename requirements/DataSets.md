# Dataset Documentation - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-04-30

---

## Overview

Two datasets for Assignment 2:

| Task | Dataset | Images | Classes | Size |
|------|---------|--------|---------|------|
| Classification | Birds (10 species) | 1,589 | 10 | ~37 MB |
| Detection | Solar Panel Damage | 1,667 | 5 | ~387 MB |

⚠️ **Important:** Use only your assigned dataset (based on student ID). Sharing datasets results in 0 marks.

---

## Image Classification Dataset

### Classes (10 bird species)

| Class | Images | Class | Images |
|-------|--------|-------|--------|
| Crested Kingfisher | 158 | Laughing Gull | 179 |
| Crow | 158 | Palila | 156 |
| Eastern Meadowlark | 173 | Paradise Tanager | 165 |
| Fairy Bluebird | 156 | Rainbow Lorikeet | 146 |
| Harlequin Quail | 139 | Townsend's Warbler | 159 |
| **Total** | **1,589** | | |

### Structure

```
data/25509225/Image_Classification/dataset/
├── CRESTED KINGFISHER/    # 158 images
├── CROW/                  # 158 images
└── ... (8 more classes)
```

**Note:** Dataset is NOT pre-split. Run `classification_split.py` to create train/valid/test splits.

---

## Object Detection Dataset

### Classes (5 damage types)

1. **Cell** - Single cell damage
2. **Cell-Multi** - Multiple cell damage
3. **No-Anomaly** - No damage detected
4. **Shadowing** - Shadow interference
5. **Unclassified** - Unknown damage type

### Formats Available

Three annotation formats provided:
- **COCO** (`coco/`) - JSON annotations
- **Pascal VOC** (`pascal/`) - XML annotations
- **YOLO** (`yolo/`) - TXT annotations with data.yaml

### Structure

```
data/25509225/Object_Detection/
├── coco/
│   ├── train/     # images + annotations.json
│   ├── valid/
│   └── test/
├── pascal/
│   ├── train/     # images + annotations.xml
│   ├── valid/
│   └── test/
└── yolo/
    ├── train/     # images/ + labels/
    ├── valid/
    └── test/
```

Total: 1,667 images across train/valid/test splits

---

## Usage

### Classification

```bash
# Split dataset first
python src/data_processing/classification_split.py

# Then use in experiments
python experiments/exp03_classification_ResNet50_v1.py
```

### Detection

```bash
# YOLOv8 (uses YOLO format)
python experiments/exp01_detection_YOLOv8.py

# Faster R-CNN (uses COCO format by default)
python experiments/exp02_detection_FasterRCNN.py
```

---

## Data Format Details

### Classification Images
- **Format:** JPEG/PNG
- **Structure:** ImageFolder (class-based folders)
- **Preprocessing:** Resize to 224x224 for ResNet50

### Detection Annotations

**COCO Format:**
```json
{
  "images": [{"id": 1, "file_name": "img.jpg", ...}],
  "annotations": [{"image_id": 1, "bbox": [x,y,w,h], "category_id": 1, ...}],
  "categories": [{"id": 1, "name": "Cell"}, ...]
}
```

**YOLO Format:**
```
0 0.5 0.5 0.2 0.3  # class x_center y_center width height (normalized)
```

**Pascal VOC:** Standard XML format with `<object>` and `<bndbox>` tags

---

## Key Notes

1. **Classification dataset requires splitting** before use (use `classification_split.py`)
2. **Detection dataset is pre-split** into train/valid/test
3. All datasets are generated uniquely per student ID
4. Do NOT share or redistribute datasets

---

**See also:**
- [Classification_Splitting.md](./Classification_Splitting.md) - How to split classification data
- [Classification_Architecture.md](./Classification_Architecture.md) - Training guide
- [Detection_Architecture.md](./Detection_Architecture.md) - Detection guide
