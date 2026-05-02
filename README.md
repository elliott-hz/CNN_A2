# CNN Assignment 2 - Object Detection & Classification

**Student ID:** 25509225  
**Course:** 42028 Deep Learning and Convolutional Neural Networks  
**Due Date:** Friday 11:59 PM, 08 May 2026

---

## 📋 Project Overview

This project implements a modular deep learning framework for **image classification** and **object detection** tasks, following assignment requirements with emphasis on **correct methodology over accuracy**.

### Key Design Principles

- ✅ **NO layer freezing** - All layers trainable from epoch 1 (teacher requirement)
- ✅ **True CNN customization** - Backbone modification (add/remove conv layers)
- ✅ **Consistent dataset splits** - Same split across experiments for fair comparison
- ✅ **Modular architecture** - Clear separation of concerns (Experiments → Models → Training → Evaluation)
- ✅ **Comprehensive tracking** - CSV metrics logging + Markdown summaries per experiment

---

## 📁 Project Structure

```
CNN_A2/
├── experiments/                    # Experiment scripts (flow control only)
│   ├── exp01_detection_YOLOv8.py          # YOLOv8 detection
│   ├── exp02_detection_FasterRCNN.py      # Faster R-CNN detection
│   ├── classification_ResNet50_baseline.py    # Baseline ResNet50
│   ├── classification_ResNet50_v1.py          # Enhanced FC head
│   ├── classification_ResNet50_v2.py          # TRUE CNN: +Conv after layer2
│   └── classification_ResNet50_v3.py          # TRUE CNN: -Layer3 (reduced)
│
├── src/                            # Core modules
│   ├── models/                     # Model definitions
│   │   ├── ResNet50ClassifierModel.py     # ResNet50 + 4 configs
│   │   ├── YOLOv8DetectorModel.py         # YOLOv8 wrapper
│   │   └── FasterRCNNDetectorModel.py     # Faster R-CNN wrapper
│   ├── training/                   # Training logic
│   │   ├── classification_trainer.py      # TrainingConfig + Trainer
│   │   ├── YOLOv8_trainer.py              # YOLOv8 trainer
│   │   └── FasterRCNN_trainer.py          # Faster R-CNN trainer
│   ├── evaluation/                 # Evaluation & visualization
│   │   ├── classification_evaluator.py    # Metrics + curves + analysis
│   │   └── detection_evaluator.py         # mAP + confusion matrix
│   ├── data_processing/            # Data loading & splitting
│   │   ├── ClassificationDataLoader.py    # Unified loader (2 aug strategies)
│   │   ├── classification_split.py        # Dataset splitter
│   │   └── faster_rcnn_dataloader.py      # Faster R-CNN dataloader
│   └── utils/                      # Utilities
│       ├── logger.py
│       └── file_utils.py
│
├── requirements/                   # Documentation
│   ├── Assignment2_Specification.md           # Official spec
│   ├── Assignment2_Specification_Teacher.md   # Teacher's emphasis
│   ├── Classification_Architecture.md         # Classification design
│   ├── Detection_Architecture.md              # Detection design
│   ├── Classification_Splitting.md            # Splitting guide
│   ├── FasterRCNN_DataLoader.md               # DataLoader guide
│   ├── DataSets.md                            # Dataset overview
│   └── LAYER_FREEZING_GUIDE.md                # Freezing reference
│
├── data/                           # Datasets (student-specific)
│   └── 25509225/
│       ├── Image_Classification/          # Birds (10 classes, 1,589 images)
│       │   ├── dataset/                   # Raw data (needs splitting)
│       │   └── split_dataset/             # Train/valid/test splits
│       └── Object_Detection/              # Solar panel damage (5 classes, 1,667 images)
│           ├── coco/                      # COCO format
│           ├── pascal/                    # Pascal VOC format
│           └── yolo/                      # YOLO format
│
└── outputs/                        # Experiment results (auto-generated)
    ├── exp01_yolov8_TIMESTAMP/
    ├── exp02_fasterrcnn_TIMESTAMP/
    ├── classification_baseline_TIMESTAMP/
    ├── classification_customized_v1_TIMESTAMP/
    ├── classification_customized_v2_TIMESTAMP/
    └── classification_customized_v3_TIMESTAMP/
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install PyTorch (choose CPU or GPU version)
# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install ultralytics>=8.0.0 scikit-learn Pillow opencv-python pyyaml pandas matplotlib seaborn
```

**Verify setup:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Prepare Datasets

Datasets are student-specific (generated based on Student ID: 25509225).

**Classification Dataset (requires splitting):**
```bash
python3 src/data_processing/classification_split.py
```
Creates train/valid/test splits in `data/25509225/Image_Classification/split_dataset/`

**Detection Dataset (pre-split):**
Already organized in `data/25509225/Object_Detection/{coco,pascal,yolo}/`

### 3. Run Experiments

#### Image Classification (4 experiments)

```bash
# Baseline: Standard ResNet50
python3 experiments/classification_ResNet50_baseline.py

# V1: Enhanced FC head (NOT true CNN customization)
python3 experiments/classification_ResNet50_v1.py

# V2: TRUE CNN customization (+Conv blocks after layer2)
python3 experiments/classification_ResNet50_v2.py

# V3: TRUE CNN customization (-Layer3, reduced depth)
python3 experiments/classification_ResNet50_v3.py
```

#### Object Detection (2 experiments)

```bash
# YOLOv8m baseline (fully implemented)
python3 experiments/exp01_detection_YOLOv8.py

# Faster R-CNN (template created, needs dataloader completion)
python3 experiments/exp02_detection_FasterRCNN.py
```

### 4. View Results

Each experiment generates a timestamped directory in `outputs/`:

```
outputs/classification_baseline_20260501_120000/
├── training/
│   ├── best_model.pth                  # Best model weights
│   └── training_history.csv            # Epoch-by-epoch metrics ⭐
├── evaluation/
│   ├── evaluation_metrics.json         # Accuracy, precision, recall, F1
│   ├── classification_report.txt       # Per-class metrics
│   └── confusion_matrix.png            # High-res heatmap ⭐
├── visualization/
│   └── training_curves.png             # Loss + accuracy plots ⭐
└── experiment_summary.md               # Comprehensive summary ⭐
```

**Key outputs:**
- 📊 **CSV files**: Detailed training history for custom analysis
- 📈 **Training curves**: Visual loss/accuracy trends
- 🔍 **Confusion matrices**: Class-wise performance analysis
- 📝 **Experiment summaries**: Complete methodology + results documentation

---

## 💻 Hardware Requirements

| Experiment | Min GPU Memory | Recommended Batch Size | Notes |
|------------|----------------|------------------------|-------|
| YOLOv8 Detection | 8GB | 16-24 | T4 optimized |
| Faster R-CNN | 10GB | 2-4 | Smaller due to two-stage |
| ResNet50 Classification | 6GB | 16 | All variants |

**If OOM errors occur:** Reduce `batch_size` in experiment script.

---

## 📊 Experiments Summary

### Image Classification

| Experiment | Model Architecture | Customization Type | Augmentation | Key Features |
|------------|-------------------|-------------------|--------------|--------------|
| **Baseline** | Standard ResNet50 | None (control) | Standard | Single FC: 2048→10 |
| **V1** | ResNet50 + Enhanced FC | FC enhancement | Enhanced | Multi-layer FC: 2048→512→256→10 + BN |
| **V2** | ResNet50 + Conv blocks | **TRUE CNN** ✓ | Enhanced | Adds conv after layer2 + enhanced FC |
| **V3** | ResNet50 - Layer3 | **TRUE CNN** ✓ | Standard | Removes layer3 (reduced depth) |

**Augmentation Strategies:**
- **Standard**: RandomRotation(15°), ColorJitter(0.2)
- **Enhanced**: RandomRotation(20°), ColorJitter(0.3+hue), RandomAffine

### Object Detection

| Experiment | Model | Status | Key Config |
|------------|-------|--------|------------|
| **Exp01** | YOLOv8m | ✅ Complete | 640×640, 100 epochs, batch=24 |
| **Exp02** | Faster R-CNN (ResNet50+FPN) | ⚠️ Template | 640×640, 50 epochs, batch=4 |

---

## 🔧 Architecture Details

### Modular Design Philosophy

```
┌─────────────────────────────────────┐
│     EXPERIMENTS (Flow Control)      │
│  - Load data                        │
│  - Initialize model                 │
│  - Configure training               │
│  - Call trainer & evaluator         │
│  - Generate summary                 │
└──────────┬──────────────────────────┘
           │
    ┌──────┼──────┬──────────┐
    ▼      ▼      ▼          ▼
┌────────┐┌──────┐┌──────┐┌──────────┐
│ MODELS ││TRAIN ││ DATA ││ EVALUATE │
│        ││      ││      ││          │
│Define  ││Loop  ││Load  ││Metrics   │
│arch    ││Optim ││Augment││Visualize │
└────────┘└──────┘└──────┘└──────────┘
```

### Centralized Configuration

All hyperparameters encapsulated in configuration objects:

**Model Configs** (`src/models/ResNet50ClassifierModel.py`):
```python
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'modify_backbone': False
}

CUSTOMIZED_V2_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.6,
    'pretrained': True,
    'additional_fc_layers': True,
    'modify_backbone': True,            # ✓ TRUE CNN customization
    'add_conv_after_layer': 'layer2'    # Add conv blocks
}
```

**Training Configs** (`src/training/classification_trainer.py`):
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    label_smoothing: float = 0.1
    use_amp: bool = True
    description: str = '...'
```

### Key Components

1. **Unified Data Loading** (`ClassificationDataLoader.py`)
   - Two augmentation strategies (standard/enhanced)
   - Eliminates ~120 lines of duplicate code
   - Ensures consistency across experiments

2. **CSV Logging** (Every epoch)
   - Tracks: train/val loss, accuracy, learning rate
   - Enables detailed post-training analysis
   - Easy import to Pandas/Excel

3. **Automated Summaries** (`ClassificationEvaluator.generate_experiment_summary()`)
   - Methodology description
   - Configuration details
   - Results + overfitting analysis
   - References to visualizations

---

## 📖 Critical Assignment Requirements

Based on teacher consultation ([Assignment2_Specification_Teacher.md](requirements/Assignment2_Specification_Teacher.md)):

### ✅ Must Do

1. **Correct Methodology > Accuracy**
   - Fair comparisons matter more than high scores
   - Document assumptions and design decisions

2. **NO Layer Freezing**
   - ❌ "If you freeze it, zero marks"
   - ✅ All layers trainable from epoch 1

3. **True CNN Customization**
   - ❌ Changing dropout/LR only ≠ customization
   - ✅ Must add/remove convolutional layers

4. **Consistent Dataset Splits**
   - Same train/val/test split across all experiments
   - Use student ID as random seed

5. **Training Curves Required**
   - Plot: train/val loss + accuracy
   - Analyze overfitting/underfitting patterns

6. **Use Provided Detection Splits**
   - ❌ Don't resplit pre-segregated detection data
   - ✅ Use existing train/valid/test folders

### ❌ Common Mistakes (Heavy Deductions)

- Freezing backbone layers
- Different splits across experiments
- Only changing hyperparameters (not CNN structure)
- Ignoring specification requirements

---

## 📝 Report Guidelines

Follow structure in [Assignment2_Specification.md](requirements/Assignment2_Specification.md):

1. **Introduction** - Overview, baseline architectures
2. **Dataset** - Description, sample images, split statistics
3. **Classification Architecture**
   - Baseline (Experiment 1)
   - Customized (Experiment 2)
   - Assumptions/intuitions
   - Model summaries
4. **Detection Architecture**
   - Faster R-CNN (Experiment 3)
   - YOLO/SSD (Experiment 4)
   - Assumptions/intuitions
   - Model summaries
5. **Experimental Results**
   - Settings (hyperparameters, augmentations)
   - Performance metrics
   - Training curves
   - Confusion matrices
   - Overfitting analysis
   - Discussion of results
6. **Conclusion** - Summary, limitations, future work

**Maximum 10 pages**

---

## 🔍 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch_size in experiment script
BATCH_SIZE = 2  # Instead of 4 or 16
```

### Training Loss Not Decreasing
- Check `model.train()` mode is set
- Verify learning rate isn't too high/low
- Ensure data is loading correctly
- Check label smoothing value

### Module Import Errors
```bash
# Use python3, not python
python3 experiments/classification_ResNet50_baseline.py

# Ensure you're in project root
cd /path/to/CNN_A2
```

### Dataset Not Found
- Verify path: `data/25509225/Image_Classification/dataset/`
- Run splitter first: `python3 src/data_processing/classification_split.py`
- Check file permissions

---

## 📚 Documentation

Detailed guides in `requirements/` folder:

- **[Classification_Architecture.md](requirements/Classification_Architecture.md)** - Complete classification system design
- **[Detection_Architecture.md](requirements/Detection_Architecture.md)** - Detection system architecture
- **[FasterRCNN_DataLoader.md](requirements/FasterRCNN_DataLoader.md)** - DataLoader implementation guide
- **[LAYER_FREEZING_GUIDE.md](requirements/LAYER_FREEZING_GUIDE.md)** - Freezing methods (reference only, not used)
- **[DataSets.md](requirements/DataSets.md)** - Dataset overview and formats
- **[Assignment2_Specification_Teacher.md](requirements/Assignment2_Specification_Teacher.md)** - Teacher's key points

---

## 🎯 Deliverables

Submit via Canvas before deadline:

1. **Report** (PDF/Word, max 10 pages)
   - Follow suggested structure
   - Include diagrams, metrics, analysis

2. **Code** (Google Colab/iPython notebooks)
   - Visible output for each cell
   - Match report results exactly

⚠️ **Both required** - Incomplete submissions won't be marked.

---

## 📞 Support

**Subject Coordinator:** Dr. Nabin Sharma  
- Email: Nabin.Sharma@uts.edu.au  
- Room: CB11.07.124  

**Questions:** Post on Canvas discussion forum (visible to all students)

---

**Last Updated:** 2026-05-02  
**Author:** Kuanlong Li (Student ID: 25509225)
