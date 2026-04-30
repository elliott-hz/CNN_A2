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

### 1. Setup GPU Environment (T4 Optimized)

```bash
chmod +x setupGPU.sh
bash setupGPU.sh
```

This installs PyTorch with CUDA and verifies GPU availability. All experiments are optimized for **NVIDIA T4 GPU (16GB)**.

### 2. Install Dependencies

```bash
pip install torch torchvision ultralytics scikit-learn Pillow
```

### 3. Run Experiments

```bash
# Object Detection (T4 optimized batch sizes)
python experiments/exp01_detection_YOLOv8.py          # batch_size=16
python experiments/exp02_detection_FasterRCNN.py      # batch_size=2

# Image Classification (T4 optimized batch sizes)
python experiments/exp03_classification_ResNet50_v1.py  # batch_size=16
python experiments/exp04_classification_ResNet50_v2.py  # batch_size=16
```

**Note:** Batch sizes are optimized for T4 GPU. If you encounter OOM errors, reduce batch_size in the experiment script.

### 4. View Results

Each experiment generates results in a structured directory:

```
outputs/
├── exp01_yolov8/
│   └── run_20260430_193045/
│       ├── training/best_model.pth
│       ├── evaluation/evaluation_metrics.json
│       └── experiment_summary.md
├── exp02_fasterrcnn/
│   └── run_YYYYMMDD_HHMMSS/
├── exp03_baseline/
│   └── run_YYYYMMDD_HHMMSS/
└── exp04_customized/
    └── run_YYYYMMDD_HHMMSS/
```

This structure allows easy comparison of multiple runs for the same experiment.

---

## 💻 Hardware Requirements

**Recommended:** NVIDIA T4 GPU (16GB VRAM) or equivalent

| Experiment | Min GPU Memory | Recommended Batch Size |
|------------|----------------|------------------------|
| YOLOv8 Detection | 8GB | 16 |
| Faster R-CNN | 10GB | 2 |
| ResNet50 Classification | 6GB | 16 |

See [T4_GPU_Optimization.md](requirements/T4_GPU_Optimization.md) for detailed optimization guide.

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
