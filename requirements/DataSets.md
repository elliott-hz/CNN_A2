# Dataset Documentation - Assignment 2

**Student ID:** 25509225  
**Student Name:** Kuanlong Li  
**Generated At:** 2026-03-30  

---

## Table of Contents

1. [Overview](#overview)
2. [Image Classification Dataset](#image-classification-dataset)
3. [Object Detection Dataset](#object-detection-dataset)
4. [Dataset Splitting Strategy](#dataset-splitting-strategy)
5. [Data Format Specifications](#data-format-specifications)
6. [Usage Guidelines](#usage-guidelines)

---

## Overview

This document provides comprehensive information about the two datasets assigned for Assignment 2 of **42028: Deep Learning and Convolutional Neural Networks**. Both datasets are uniquely generated based on student ID **25509225** to ensure individual assessment integrity.

### Dataset Summary

| Dataset Type | Dataset Name | Total Images | Classes/Categories | Storage Size |
|-------------|--------------|--------------|-------------------|--------------|
| Image Classification | Birds (10 species) | 1,589 | 10 classes | ~37 MB |
| Object Detection | Solar Panel Damage Detection | 1,667 | 5 categories | ~387 MB |

### Important Notes

⚠️ **Critical Requirements:**
- Only use the dataset assigned to your student ID
- Any discrepancy will result in **0 marks** for the entire assignment
- The same dataset will always be generated for a given student ID
- Datasets must NOT be shared or reproduced outside academic purposes

---

## Image Classification Dataset

### Dataset Information

**Dataset Name:** Birds Image Classification  
**Source:** Selected from publicly available bird image datasets  
**License:** Academic use only (UTS modified version)  

### Class Distribution

The dataset contains **10 unique bird species** with varying image counts per class:

| Class ID | Class Name | Image Count | Percentage |
|----------|-----------|-------------|------------|
| 1 | CRESTED KINGFISHER | 158 | 9.94% |
| 2 | CROW | 158 | 9.94% |
| 3 | EASTERN MEADOWLARK | 173 | 10.89% |
| 4 | FAIRY BLUEBIRD | 156 | 9.82% |
| 5 | HARLEQUIN QUAIL | 139 | 8.75% |
| 6 | LAUGHING GULL | 179 | 11.27% |
| 7 | PALILA | 156 | 9.82% |
| 8 | PARADISE TANAGER | 165 | 10.38% |
| 9 | RAINBOW LORIKEET | 146 | 9.19% |
| 10 | TOWNSENDS WARBLER | 159 | 10.01% |
| **Total** | **-** | **1,589** | **100%** |

### Directory Structure

```
data/25509225/Image_Classification/
└── dataset/
    ├── CRESTED KINGFISHER/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ... (158 images total)
    ├── CROW/
    │   ├── image_001.jpg
    │   └── ... (158 images total)
    ├── EASTERN MEADOWLARK/
    │   └── ... (173 images total)
    ├── FAIRY BLUEBIRD/
    │   └── ... (156 images total)
    ├── HARLEQUIN QUAIL/
    │   └── ... (139 images total)
    ├── LAUGHING GULL/
    │   └── ... (179 images total)
    ├── PALILA/
    │   └── ... (156 images total)
    ├── PARADISE TANAGER/
    │   └── ... (165 images total)
    ├── RAINBOW LORIKEET/
    │   └── ... (146 images total)
    └── TOWNSENDS WARBLER/
        └── ... (159 images total)
```

### Key Characteristics

- **Format:** JPEG/PNG images
- **Organization:** Class-based folder structure (ImageNet-style)
- **Pre-split:** ❌ **NO** - Dataset is NOT pre-split into train/test/validation
- **Splitting Required:** ✅ Students MUST split the dataset themselves
- **Random Seed:** Use student ID (**25509225**) as the random seed for reproducibility
- **Image Properties:** 
  - Various resolutions (typical range: 200x200 to 1024x1024 pixels)
  - Color images (RGB)
  - Natural backgrounds with birds in various poses

### Challenges & Considerations

1. **Class Imbalance:** Slight variation in image counts (139-179 per class)
2. **Intra-class Variation:** Birds may appear in different poses, lighting conditions, and backgrounds
3. **Inter-class Similarity:** Some bird species may share visual similarities
4. **Background Complexity:** Natural environments with varying complexity

---

## Object Detection Dataset

### Dataset Information

**Dataset Name:** Solar Panel Damage Detection - Thermal Image Dataset  
**Original Creator:** Rinat Landman  
**License:** CC BY 4.0 (original) | Modified for UTS Academic Purposes  
**Description:** Over 1,500 thermal labeled images of solar panels for object detection  

### Annotation Categories

The dataset contains **5 object categories** related to solar panel damage detection:

| Category ID | Category Name | Description |
|-------------|--------------|-------------|
| 1 | Cell | Individual solar cell anomalies |
| 2 | Cell-Multi | Multiple cell anomalies in clusters |
| 3 | No-Anomaly | Normal/healthy solar panel regions |
| 4 | Shadowing | Shadow effects on panels |
| 5 | Unclassified | Anomalies that don't fit other categories |

### Data Formats Provided

The dataset is provided in **three annotation formats** for flexibility:

1. **COCO Format** (Common Objects in Context)
2. **Pascal VOC Format** (XML annotations)
3. **YOLO Format** (Compatible with YOLOv5/v8)

### Dataset Splitting

✅ **Pre-split** - The dataset is already segregated into train/valid/test sets:

| Split | Image Count | Percentage | Annotations |
|-------|-------------|------------|-------------|
| **Train** | 1,167 | 70% | ~63,554 |
| **Validation** | 250 | 15% | ~12,915 |
| **Test** | 250 | 15% | ~15,083 |
| **Total** | **1,667** | **100%** | **~91,552** |

**Split Ratios:** Train: 70%, Valid: 15%, Test: 15%

### Directory Structures

#### 1. COCO Format Structure

```
data/25509225/Object_Detection/coco/
├── README.txt
├── train/
│   ├── 100001.jpg
│   ├── 100002.jpg
│   ├── ... (1,167 images)
│   └── train_annotations.json
├── valid/
│   ├── [validation images]
│   └── valid_annotations.json
└── test/
    ├── [test images]
    └── test_annotations.json
```

**COCO Annotation File Structure:**
```json
{
  "images": [...],      // List of image metadata
  "annotations": [...], // List of bounding box annotations
  "categories": [...]   // List of 5 category definitions
}
```

#### 2. Pascal VOC Format Structure

```
data/25509225/Object_Detection/pascal/
├── README.txt
├── train/
│   ├── 100001.jpg
│   ├── 100001.xml      // Corresponding annotation
│   ├── 100002.jpg
│   ├── 100002.xml
│   └── ... (1,167 image-annotation pairs)
├── valid/
│   ├── [250 image-xml pairs]
└── test/
    ├── [250 image-xml pairs]
```

**Pascal XML Annotation Example:**
```xml
<annotation>
  <filename>100001.jpg</filename>
  <object>
    <name>Cell</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
  <!-- More objects... -->
</annotation>
```

#### 3. YOLO Format Structure

```
data/25509225/Object_Detection/yolo/
├── README.txt
├── data.yaml           // YOLO configuration file
├── train/
│   ├── images/
│   │   ├── 100001.jpg
│   │   └── ... (1,167 images)
│   └── labels/
│       ├── 100001.txt  // Corresponding annotation
│       └── ... (1,167 label files)
├── valid/
│   ├── images/
│   │   └── [250 images]
│   └── labels/
│       └── [250 label files]
└── test/
    ├── images/
    │   └── [250 images]
    └── labels/
        └── [250 label files]
```

**YOLO data.yaml Configuration:**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']
```

**YOLO Label File Format (.txt):**
```
<class_id> <x_center> <y_center> <width> <height>
```
- Values are normalized (0-1 range) relative to image dimensions
- One line per object in the image

### Key Characteristics

- **Image Type:** Thermal imaging (infrared)
- **Application:** Solar panel damage/fault detection
- **Annotation Density:** High (~55 annotations per image on average)
- **Object Sizes:** Varying scales from small cell defects to large shadowing areas
- **Image Properties:**
  - Grayscale thermal images
  - Resolution varies (typically 400x400 to 800x800 pixels)
  - High contrast between anomalies and normal regions

### Challenges & Considerations

1. **High Annotation Density:** Many objects per image require efficient processing
2. **Small Object Detection:** Cell-level defects may be very small relative to image size
3. **Class Imbalance:** Some categories (e.g., No-Anomaly) may dominate
4. **Thermal Image Specifics:** Different from RGB - requires understanding of thermal patterns
5. **Overlap:** Multiple anomalies may overlap or be adjacent

---

## Dataset Splitting Strategy

### Image Classification: Manual Splitting Required

Since the Image Classification dataset is **NOT pre-split**, you must create your own train/validation/test splits.

#### Recommended Approach

```python
import os
import shutil
import random
from pathlib import Path

# Configuration
STUDENT_ID = 25509225
DATASET_PATH = "data/25509225/Image_Classification/dataset"
OUTPUT_PATH = "data/25509225/Image_Classification/split_dataset"

# Set random seed for reproducibility
random.seed(STUDENT_ID)

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset():
    """Split the dataset into train/valid/test using student ID as seed."""
    
    classes = sorted(os.listdir(DATASET_PATH))
    
    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle with fixed seed
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_valid = int(n_total * VALID_RATIO)
        
        train_imgs = images[:n_train]
        valid_imgs = images[n_train:n_train + n_valid]
        test_imgs = images[n_train + n_valid:]
        
        # Create directories
        for split, imgs in [('train', train_imgs), 
                           ('valid', valid_imgs), 
                           ('test', test_imgs)]:
            split_dir = os.path.join(OUTPUT_PATH, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            
            # Copy images
            for img in imgs:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
        
        print(f"{cls}: Train={len(train_imgs)}, Valid={len(valid_imgs)}, Test={len(test_imgs)}")

if __name__ == "__main__":
    split_dataset()
```

#### Expected Split Results

Based on the dataset statistics:

| Class | Total | Train (70%) | Valid (15%) | Test (15%) |
|-------|-------|-------------|-------------|------------|
| CRESTED KINGFISHER | 158 | 110 | 23 | 25 |
| CROW | 158 | 110 | 23 | 25 |
| EASTERN MEADOWLARK | 173 | 121 | 25 | 27 |
| FAIRY BLUEBIRD | 156 | 109 | 23 | 24 |
| HARLEQUIN QUAIL | 139 | 97 | 20 | 22 |
| LAUGHING GULL | 179 | 125 | 26 | 28 |
| PALILA | 156 | 109 | 23 | 24 |
| PARADISE TANAGER | 165 | 115 | 24 | 26 |
| RAINBOW LORIKEET | 146 | 102 | 21 | 23 |
| TOWNSENDS WARBLER | 159 | 111 | 23 | 25 |
| **TOTAL** | **1,589** | **1,109** | **231** | **249** |

**Note:** Exact numbers may vary slightly due to rounding.

### Object Detection: Pre-split (Use As-Is)

The Object Detection dataset is **already split**. Simply use the provided structure:

- **Training:** Use `train/` directory (1,167 images)
- **Validation:** Use `valid/` directory (250 images)
- **Testing:** Use `test/` directory (250 images)

**No additional splitting required!**

---

## Data Format Specifications

### Image Classification

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)

**Recommended PyTorch DataLoader Structure:**

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(
    root='data/25509225/Image_Classification/split_dataset/train',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### Object Detection

#### COCO Format Usage

**For TorchVision Faster R-CNN:**

```python
from torchvision.datasets import CocoDetection

train_dataset = CocoDetection(
    root='data/25509225/Object_Detection/coco/train',
    annFile='data/25509225/Object_Detection/coco/train/train_annotations.json'
)
```

#### Pascal VOC Format Usage

**For custom parsers:**

```python
import xml.etree.ElementTree as ET

def parse_pascal_xml(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return objects
```

#### YOLO Format Usage

**For Ultralytics YOLOv8:**

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train with YOLO format dataset
results = model.train(
    data='data/25509225/Object_Detection/yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## Usage Guidelines

### Best Practices

1. **Reproducibility:**
   - Always use student ID (25509225) as random seed
   - Document all preprocessing steps
   - Save data splitting code in your notebook

2. **Data Exploration:**
   - Visualize sample images from each class/category
   - Check for corrupted or missing images
   - Analyze class distribution and balance

3. **Preprocessing:**
   - Apply appropriate data augmentation
   - Normalize images consistently
   - Handle varying image sizes appropriately

4. **Memory Management:**
   - Image Classification: ~37 MB (manageable in memory)
   - Object Detection: ~387 MB (consider batch loading for large models)

### Common Pitfalls to Avoid

❌ **Don't:**
- Use a different dataset than assigned
- Share your dataset with others
- Forget to set the random seed
- Skip data validation checks
- Mix up annotation formats

✅ **Do:**
- Verify dataset integrity before training
- Document your data splitting strategy
- Include sample visualizations in your report
- Test with small subsets first
- Back up your processed data

### Quality Checks

Run these checks before starting experiments:

```python
import os
from PIL import Image

def verify_image_classification_dataset():
    """Verify all images can be loaded."""
    base_path = "data/25509225/Image_Classification/dataset"
    errors = []
    
    for cls in os.listdir(base_path):
        cls_path = os.path.join(base_path, cls)
        if not os.path.isdir(cls_path):
            continue
        
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's a valid image
            except Exception as e:
                errors.append(f"Error in {img_path}: {str(e)}")
    
    if errors:
        print(f"Found {len(errors)} problematic images:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
    else:
        print("✅ All images verified successfully!")
    
    return len(errors) == 0

# Run verification
verify_image_classification_dataset()
```

---

## Technical Specifications

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Image Classification Training | CPU, 8GB RAM | GPU (4GB+ VRAM), 16GB RAM |
| Object Detection Training | GPU (4GB VRAM), 16GB RAM | GPU (8GB+ VRAM), 32GB RAM |
| Inference (Both) | CPU, 4GB RAM | GPU (2GB+ VRAM), 8GB RAM |

### Software Requirements

- **Python:** 3.8+
- **PyTorch:** 2.0+
- **Torchvision:** 0.15+
- **Ultralytics:** 8.0+ (for YOLO)
- **OpenCV:** 4.7+
- **Pillow:** 9.5+
- **NumPy:** >=1.24.0, <2.0.0

### Storage Requirements

| Component | Space Required |
|-----------|---------------|
| Raw Datasets | ~424 MB |
| Processed/Split Data | ~500 MB (estimated) |
| Model Weights | 500 MB - 2 GB (depending on architecture) |
| Training Outputs | 1-5 GB (logs, checkpoints, visualizations) |
| **Total Recommended** | **5-10 GB free space** |

---

## Troubleshooting

### Common Issues

**Issue 1: Dataset path not found**
```
Solution: Verify the path structure matches:
  data/25509225/Image_Classification/dataset/
  data/25509225/Object_Detection/[coco|pascal|yolo]/
```

**Issue 2: Corrupted images**
```
Solution: Run the verification script above and remove/re-download problematic images.
```

**Issue 3: Memory errors during training**
```
Solution: 
  - Reduce batch size
  - Use smaller image resolutions
  - Enable gradient checkpointing
  - Use mixed precision training (AMP)
```

**Issue 4: Annotation format mismatch**
```
Solution: Ensure you're using the correct parser for your chosen format:
  - COCO: Use torchvision.datasets.CocoDetection
  - Pascal: Custom XML parser or detectron2
  - YOLO: Ultralytics YOLO framework
```

---

## References & Resources

### Official Documentation

- [PyTorch Vision Datasets](https://pytorch.org/vision/stable/datasets.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)
- [YOLO Documentation](https://docs.ultralytics.com/)

### Dataset Attribution

- **Birds Dataset:** Compiled from publicly available sources, modified for UTS academic purposes
- **Solar Panel Dataset:** Originally created by Rinat Landman, licensed under CC BY 4.0, modified for UTS academic purposes

### Contact

For dataset-related issues:
- Post questions on Canvas discussion forum
- Contact Subject Coordinator: Dr. Nabin Sharma (Nabin.Sharma@uts.edu.au)

---

**Last Updated:** 2026-04-30  
**Document Version:** 1.0  
**Author:** Kuanlong Li (Student ID: 25509225)

---

*This document is for academic purposes only as part of Assignment 2 for 42028: Deep Learning and Convolutional Neural Networks at University of Technology Sydney.*
