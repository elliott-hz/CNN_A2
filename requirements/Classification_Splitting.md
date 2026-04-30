# Image Classification Dataset Splitting Guide

**Student ID:** 25509225  
**Last Updated:** 2026-04-30

---

## Quick Start

```bash
python src/data_processing/classification_split.py
```

This script:
- Reads dataset from `data/25509225/Image_Classification/dataset/`
- Splits into train (70%), valid (15%), test (15%)
- Saves to `data/25509225/Image_Classification/split_dataset/`
- Generates JSON report in `outputs/`

---

## Output Structure

```
split_dataset/
├── train/    # ~1,109 images (70%)
├── valid/    # ~231 images (15%)
└── test/     # ~249 images (15%)
```

Each split contains 10 class subdirectories (ImageFolder format).

---

## Usage with PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transforms for ResNet50
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('data/25509225/Image_Classification/split_dataset/train', transform=train_transform)
valid_dataset = datasets.ImageFolder('data/25509225/Image_Classification/split_dataset/valid', transform=test_transform)
test_dataset = datasets.ImageFolder('data/25509225/Image_Classification/split_dataset/test', transform=test_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```

---

## Key Features

- **Reproducible**: Uses student ID (25509225) as random seed
- **Stratified**: Maintains class distribution across splits
- **Verified**: Automatically checks split integrity
- **Reported**: Generates JSON statistics report

---

## Expected Statistics

| Split | Images | Percentage |
|-------|--------|------------|
| Train | ~1,109 | 70% |
| Valid | ~231 | 15% |
| Test | ~249 | 15% |
| **Total** | **1,589** | **100%** |

10 classes: Crested Kingfisher, Crow, Eastern Meadowlark, Fairy Bluebird, Harlequin Quail, Laughing Gull, Palila, Paradise Tanager, Rainbow Lorikeet, Townsend's Warbler

---

## Troubleshooting

**Dataset not found**: Ensure data exists at `data/25509225/Image_Classification/dataset/`

**No images found**: Check file extensions (.jpg, .png, etc.) and verify files aren't corrupted

**Permission denied**: Ensure write access to output directory

---

**See also:** [Classification_Architecture.md](./Classification_Architecture.md) for model training details
