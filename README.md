# CNN Assignment 2 - Object Detection & Classification

**Student ID:** 25509225  
**Course:** 42028 Deep Learning and Convolutional Neural Networks

---

## 📋 Project Structure

```
CNN_A2/
├── experiments/                    # Experiment scripts
│   ├── exp01_detection_YOLOv8.py          # YOLOv8 detection
│   ├── exp02_detection_FasterRCNN.py      # Faster R-CNN detection
│   ├── exp03_classification_ResNet50_v1.py  # ResNet50 baseline classification
│   └── exp04_classification_ResNet50_v2.py  # ResNet50 customized classification
│
├── src/
│   ├── models/                     # Model definitions
│   │   ├── ResNet50ClassifierModel.py
│   │   ├── YOLOv8DetectorModel.py
│   │   └── FasterRCNNDetectorModel.py
│   ├── training/                   # Training modules
│   │   ├── classification_trainer.py
│   │   ├── YOLOv8_trainer.py
│   │   └── FasterRCNN_trainer.py
│   ├── evaluation/                 # Evaluation modules
│   │   ├── classification_evaluator.py
│   │   └── detection_evaluator.py
│   └── data_processing/            # Data processing
│       ├── classification_split.py
│       └── faster_rcnn_dataloader.py
│
├── requirements/                   # Documentation
│   ├── Classification_Architecture.md
│   ├── Detection_Architecture.md
│   ├── Classification_Splitting.md
│   ├── FasterRCNN_DataLoader.md
│   └── DataSets.md
│
└── outputs/                        # Experiment results (auto-generated)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision ultralytics scikit-learn Pillow
```

### 2. Run Experiments

```bash
# Object Detection
python experiments/exp01_detection_YOLOv8.py
python experiments/exp02_detection_FasterRCNN.py

# Image Classification
python experiments/exp03_classification_ResNet50_v1.py
python experiments/exp04_classification_ResNet50_v2.py
```

### 3. View Results

Each experiment generates a timestamped directory in `outputs/` containing:
- `training/best_model.pth` - Best model weights
- `evaluation/evaluation_metrics.json` - Evaluation metrics
- `experiment_summary.md` - Experiment summary (Markdown format)

---

## 📊 Experiments Overview

| Experiment | Model | Task | Status |
|------------|-------|------|--------|
| Exp01 | YOLOv8m | Object Detection | ✅ Complete |
| Exp02 | Faster R-CNN | Object Detection | ✅ Complete |
| Exp03 | ResNet50 Baseline | Image Classification | ✅ Complete |
| Exp04 | ResNet50 Customized | Image Classification | ✅ Complete |

---

## 📖 Detailed Documentation

- **[Classification_Architecture.md](requirements/Classification_Architecture.md)** - Classification task architecture
- **[Detection_Architecture.md](requirements/Detection_Architecture.md)** - Detection task architecture
- **[Classification_Splitting.md](requirements/Classification_Splitting.md)** - Dataset splitting guide
- **[FasterRCNN_DataLoader.md](requirements/FasterRCNN_DataLoader.md)** - DataLoader implementation
- **[DataSets.md](requirements/DataSets.md)** - Dataset overview

---

## 🔧 Architecture Design Principles

1. **Modularity**: Models, training, and evaluation are completely separated
2. **Simplicity**: Each file has a single responsibility
3. **Reproducibility**: Fixed random seeds, embedded configurations
4. **Maintainability**: Clear code structure and documentation

---

**Last Updated:** 2026-04-30
