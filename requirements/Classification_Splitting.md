# Image Classification Dataset Splitting Guide

**Student ID:** 25509225  
**Document Version:** 1.0  
**Last Updated:** 2026-04-30  

---

## Overview

This document explains how to split the Image Classification dataset for use with **ResNet50** and custom CNN architectures. The splitting script ensures reproducibility by using the student ID as a random seed.

---

## Quick Start

### 1. Run the Splitting Script

```bash
python src/data_processing/classification_split.py
```

The script will:
- Read the original dataset from `data/25509225/Image_Classification/dataset/`
- Split it into train (70%), valid (15%), test (15%)
- Save the split dataset to `data/25509225/Image_Classification/split_dataset/`
- Generate a JSON report in `outputs/`

### 2. Verify the Output

After running, you should see output like:

```
================================================================================
IMAGE CLASSIFICATION DATASET SPLITTING
================================================================================

Student ID: 25509225
Original Dataset: data/25509225/Image_Classification/dataset
Output Directory: data/25509225/Image_Classification/split_dataset
Split Ratios - Train: 70%, Valid: 15%, Test: 15%

--------------------------------------------------------------------------------
Found 10 classes:
  1. CRESTED KINGFISHER
  2. CROW
  3. EASTERN MEADOWLARK
  ...

--------------------------------------------------------------------------------
Processing each class...
--------------------------------------------------------------------------------

✓ CRESTED KINGFISHER            | Total:  158 | Train:  110 | Valid:   23 | Test:   25
✓ CROW                          | Total:  158 | Train:  110 | Valid:   23 | Test:   25
...

================================================================================
SPLITTING SUMMARY
================================================================================

Total Images: 1589
  Training:   1109 (69.8%)
  Validation: 231 (14.5%)
  Testing:    249 (15.7%)

Output saved to: data/25509225/Image_Classification/split_dataset
================================================================================

✅ Verification PASSED - All splits are valid!

================================================================================
✅ DATASET SPLITTING COMPLETED SUCCESSFULLY!
================================================================================

You can now use the split dataset for training:
  Training:   data/25509225/Image_Classification/split_dataset/train/
  Validation: data/25509225/Image_Classification/split_dataset/valid/
  Testing:    data/25509225/Image_Classification/split_dataset/test/

The dataset is compatible with torchvision.datasets.ImageFolder
================================================================================
```

---

## Directory Structure

### Before Splitting

```
data/25509225/Image_Classification/
└── dataset/                    # Original unsplit dataset
    ├── CRESTED KINGFISHER/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── CROW/
    │   └── ...
    └── ... (8 more classes)
```

### After Splitting

```
data/25509225/Image_Classification/
├── dataset/                    # Original (unchanged)
│   └── ...
└── split_dataset/              # NEW: Split dataset
    ├── train/                  # 70% of data (~1109 images)
    │   ├── CRESTED KINGFISHER/
    │   │   ├── image_XXX.jpg
    │   │   └── ...
    │   ├── CROW/
    │   │   └── ...
    │   └── ... (all 10 classes)
    ├── valid/                  # 15% of data (~231 images)
    │   ├── CRESTED KINGFISHER/
    │   │   └── ...
    │   └── ... (all 10 classes)
    └── test/                   # 15% of data (~249 images)
        ├── CRESTED KINGFISHER/
        │   └── ...
        └── ... (all 10 classes)
```

---

## Compatibility with ResNet50 and Custom CNNs

The split dataset follows the **torchvision.datasets.ImageFolder** format, making it directly compatible with standard PyTorch workflows.

### Loading the Dataset with PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for ResNet50
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='data/25509225/Image_Classification/split_dataset/train',
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    root='data/25509225/Image_Classification/split_dataset/valid',
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root='data/25509225/Image_Classification/split_dataset/test',
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Class names: {train_dataset.classes}")
```

### Using with ResNet50

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify final layer for 10 classes
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')
```

---

## Configuration Options

### Customizing Split Ratios

You can modify the split ratios by editing the script:

```python
splitter = ClassificationDatasetSplitter(
    student_id="25509225",
    train_ratio=0.80,  # Change to 80%
    valid_ratio=0.10,  # Change to 10%
    test_ratio=0.10    # Change to 10%
)
```

**Note:** The ratios must sum to 1.0 (or very close to it).

### Using Different Student ID

If needed, you can change the student ID (though this is not recommended for Assignment 2):

```python
splitter = ClassificationDatasetSplitter(student_id="YOUR_STUDENT_ID")
```

---

## Expected Statistics

Based on the current dataset (1,589 images across 10 classes), here are the expected split results:

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

*Note: Exact numbers may vary slightly due to integer rounding.*

---

## Reproducibility

The splitting process is **fully reproducible** because:

1. **Fixed Random Seed:** Uses student ID (25509225) as the random seed
2. **Deterministic Shuffling:** Same seed always produces the same split
3. **Sorted Classes:** Classes are processed in alphabetical order

Running the script multiple times will produce **identical results**.

---

## Verification

The script automatically verifies the split integrity by checking:

- ✅ All split directories exist
- ✅ Each class has images in all splits
- ✅ Sample images can be opened without errors
- ✅ No corrupted files

If verification fails, the script will report specific errors.

---

## Report Generation

After splitting, a JSON report is automatically generated in the `outputs/` directory:

```
outputs/
└── classification_split_report_YYYYMMDD_HHMMSS.json
```

### Report Contents

```json
{
  "timestamp": "2026-04-30T17:35:03.123456",
  "student_id": "25509225",
  "split_ratios": {
    "train": 0.7,
    "valid": 0.15,
    "test": 0.15
  },
  "statistics": {
    "classes": {
      "CRESTED KINGFISHER": {
        "total": 158,
        "train": 110,
        "valid": 23,
        "test": 25
      },
      ...
    },
    "totals": {
      "train": 1109,
      "valid": 231,
      "test": 249,
      "total": 1589
    }
  },
  "output_path": "data/25509225/Image_Classification/split_dataset"
}
```

This report can be used for:
- Documenting your experimental setup
- Verifying split consistency across runs
- Including in your assignment report

---

## Troubleshooting

### Issue 1: "Original dataset not found"

**Error:**
```
FileNotFoundError: Original dataset not found at: data/25509225/Image_Classification/dataset
```

**Solution:**
- Verify that the dataset exists at the correct path
- Check that you're running the script from the project root directory
- Ensure the dataset was properly downloaded/generated

### Issue 2: "No images found for class"

**Warning:**
```
⚠ Warning: No images found for class 'CLASS_NAME', skipping...
```

**Solution:**
- Check if the class directory contains image files
- Verify file extensions (.jpg, .jpeg, .png, .bmp, .tiff)
- Ensure files are not corrupted

### Issue 3: "Corrupted image"

**Error during verification:**
```
Corrupted image: path/to/image.jpg - cannot identify image file
```

**Solution:**
- Remove or replace the corrupted image
- Re-run the splitting script
- Use image validation tools to check dataset integrity

### Issue 4: Memory Error

**Error:**
```
MemoryError: Unable to allocate memory
```

**Solution:**
- Close other applications
- Reduce batch size in subsequent training
- Consider using a machine with more RAM

### Issue 5: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
- Check file permissions
- Ensure you have write access to the output directory
- Run with appropriate user privileges

---

## Best Practices

### 1. Always Verify the Split

After running the script, check the verification output:

```bash
✅ Verification PASSED - All splits are valid!
```

If it fails, investigate the reported errors before proceeding.

### 2. Keep the Original Dataset Intact

The script does NOT modify the original dataset. Always keep `dataset/` as backup.

### 3. Document Your Split

Save the generated JSON report and include it in your assignment submission to prove reproducibility.

### 4. Visualize Class Distribution

Before training, visualize the class distribution to check for imbalance:

```python
import matplotlib.pyplot as plt
from collections import Counter

# Count samples per class
class_counts = Counter([label for _, label in train_dataset.samples])

# Plot
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Training Set Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
```

### 5. Use Consistent Transforms

Ensure training and testing use appropriate transforms:
- **Training:** Include augmentation (flip, rotation, etc.)
- **Testing:** Only resize, crop, and normalize (no augmentation)

---

## Integration with Experiment Scripts

### Example: Integrating with ResNet50 Experiment

```python
# In your experiment script (e.g., experiments/exp_classification_resnet50.py)

from src.data_processing.classification_split import ClassificationDatasetSplitter

# Step 1: Split dataset (run once)
if not Path("data/25509225/Image_Classification/split_dataset").exists():
    print("Splitting dataset...")
    splitter = ClassificationDatasetSplitter(student_id="25509225")
    splitter.perform_split()
    splitter.generate_split_report()

# Step 2: Load split dataset
train_dataset = datasets.ImageFolder(
    root='data/25509225/Image_Classification/split_dataset/train',
    transform=train_transform
)

# ... continue with training
```

---

## Advanced Usage

### Programmatic Access

You can also use the splitter programmatically in your code:

```python
from src.data_processing import ClassificationDatasetSplitter

# Initialize
splitter = ClassificationDatasetSplitter(
    student_id="25509225",
    train_ratio=0.70,
    valid_ratio=0.15,
    test_ratio=0.15
)

# Perform split
stats = splitter.perform_split()

# Access statistics
print(f"Training images: {stats['totals']['train']}")
print(f"Classes: {list(stats['classes'].keys())}")

# Verify
is_valid = splitter.verify_split_integrity()
```

### Custom Image Extensions

To support additional image formats, modify the `image_extensions` set:

```python
self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
```

---

## Performance Tips

1. **Use SSD Storage:** Faster I/O for large datasets
2. **Enable Pin Memory:** Set `pin_memory=True` in DataLoader for faster GPU transfer
3. **Adjust Workers:** Set `num_workers` based on your CPU cores (typically 4-8)
4. **Pre-fetch Data:** Use larger batch sizes if memory allows

---

## Related Documentation

- [DataSets.md](./DataSets.md) - Complete dataset documentation
- [Assignment2_Specification.md](./Assignment2_Specification.md) - Assignment requirements
- [PyTorch ImageFolder Documentation](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder)

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the error messages carefully
- Post on Canvas discussion forum
- Contact: Dr. Nabin Sharma (Nabin.Sharma@uts.edu.au)

---

**Author:** Kuanlong Li (Student ID: 25509225)  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Institution:** University of Technology Sydney  

*This document is for academic purposes only.*
