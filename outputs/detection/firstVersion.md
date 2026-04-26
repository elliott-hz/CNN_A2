# Detection Model Comparison Experiments

This directory contains three detection model comparison experiments for dog face detection.

## Overview

| Experiment | Model | Architecture | Speed | Accuracy |
|------------|-------|--------------|-------|----------|
| Exp01 | YOLOv8-Medium | Single-stage | Fast | Good |
| Exp02 | Faster R-CNN | Two-stage (ResNet50+FPN) | Slow | Best |
| Exp03 | SSD300 | Single-stage (VGG16) | Fastest | Moderate |

## Prerequisites

All experiments require preprocessed data in the correct format:

```bash
# Step 1: Preprocess raw data to YOLO format
python src/data_processing/detection_preprocessor.py

# Step 2: Convert to COCO and VOC formats
python src/data_processing/convert_detection_format.py
```

This creates:
- `data/processed/detection/` - YOLO format (for Exp01)
- `data/processed/detection_coco/` - COCO JSON format (for Exp02 & Exp03)
- `data/processed/detection_voc/` - VOC XML format (alternative for Exp03)

## Running Experiments

### Experiment 01: YOLOv8 Baseline
```bash
python experiments/exp01_detection_YOLOv8_baseline.py
```

**Configuration:**
- Backbone: Medium (YOLOv8m)
- Input size: 640x640
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 24
- Epochs: 120

### Experiment 02: Faster R-CNN
```bash
python experiments/exp02_detection_FasterRCNN.py
```

**Configuration:**
- Backbone: ResNet50 + FPN
- Input size: 800-1333 (multi-scale)
- Optimizer: SGD with momentum
- Learning rate: 0.005
- Batch size: 4 (effective 8 with gradient accumulation)
- Epochs: 120

**Key Features:**
- Two-stage detection (RPN + ROI pooling)
- Highest accuracy among the three
- Slower inference speed
- Better for small objects

### Experiment 03: SSD
```bash
python experiments/exp03_detection_SSD.py
```

**Configuration:**
- Backbone: VGG16
- Input size: 300x300
- Optimizer: SGD with momentum
- Learning rate: 0.002
- Batch size: 8
- Epochs: 120

**Key Features:**
- Single-stage detection
- Fastest inference speed
- Multi-scale feature maps
- Good balance of speed and accuracy

## Common Arguments

All experiments support:
- `--resume`: Resume training from the latest checkpoint

Example:
```bash
python experiments/exp02_detection_FasterRCNN.py --resume
```

## Output Structure

Each experiment creates an output directory:
```
outputs/expXX_detection_*/
├── run_YYYYMMDD_HHMMSS/
│   ├── model/
│   │   └── best_model.pt
│   ├── logs/
│   │   ├── training.log
│   │   └── training_log.csv
│   ├── evaluation_results.yaml
│   └── EXPERIMENT_SUMMARY.md
```

## Expected Results Comparison

Based on typical performance characteristics:

| Metric | YOLOv8-M | Faster R-CNN | SSD300 |
|--------|----------|--------------|--------|
| mAP@0.5 | ~0.75-0.85 | ~0.80-0.90 | ~0.70-0.80 |
| mAP@0.5:0.95 | ~0.50-0.60 | ~0.55-0.65 | ~0.45-0.55 |
| Inference FPS | ~30-50 | ~5-10 | ~40-60 |
| Training Time | Medium | Slow | Fast |
| Memory Usage | Medium | High | Low |

*Note: Actual results depend on dataset quality, hyperparameters, and hardware.*

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or enable gradient accumulation:
```python
training_config = {
    'batch_size': 2,  # Reduce this
    'gradient_accumulation_steps': 4,  # Increase this
    # ... other configs
}
```

### Low mAP
1. Check data quality (verify labels are correct)
2. Increase training epochs
3. Adjust learning rate
4. Try different augmentations

### Slow Training
1. Enable AMP (already enabled by default)
2. Increase num_workers in dataloaders
3. Use smaller input sizes (for SSD)

## Implementation Details

### Data Format Conversion
- **YOLO → COCO**: Normalized coordinates converted to pixel coordinates
- **COCO → VOC**: Bounding boxes converted from [x,y,w,h] to [x1,y1,x2,y2]
- All formats maintain the same train/val/test split

### Training Framework
- **YOLOv8**: Uses Ultralytics built-in trainer
- **Faster R-CNN & SSD**: Custom manual training loop with PyTorch native APIs
- All use cosine annealing scheduler with warmup
- Mixed precision training (AMP) enabled for all models

### Evaluation Metrics
- mAP@0.5: IoU threshold = 0.5
- mAP@0.5:0.95: Average over IoU thresholds 0.5 to 0.95
- Calculated using custom implementation (can be enhanced with pycocotools)

## Future Improvements

1. **Enhanced mAP Calculation**: Integrate pycocotools for more accurate metrics
2. **Data Augmentation**: Add Mosaic, MixUp for YOLOv8; RandomHorizontalFlip for torchvision models
3. **Hyperparameter Tuning**: Use Optuna or Ray Tune for automatic tuning
4. **Model Export**: Add ONNX export for deployment
5. **Visualization**: Add prediction visualization and confusion matrices

## References

- YOLOv8: https://docs.ultralytics.com/
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- SSD: https://arxiv.org/abs/1512.02325
- Torchvision Detection: https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection