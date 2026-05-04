"""
Experiment V3: YOLOv8 with Shallower Backbone (Reduced Convolutional Layers)

This experiment removes convolutional layers from the YOLOv8 backbone:
- Reduces C2f module repetitions in layer4 (6→3) and layer5 (3→2)
- Removes 8 convolutional layers total
- Tests if a lighter model can maintain performance with faster inference

Customization Details:
- Location: Backbone layer4 and layer5
- Removed layers:
  * Reduced layer4 C2f from 6 to 3 repeats (removed 3 bottlenecks = 6 conv layers)
  * Reduced layer5 C2f from 3 to 2 repeats (removed 1 bottleneck = 2 conv layers)
- Total removed: 8 convolutional layers
- Expected parameter decrease: ~12-15%
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
import yaml
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_V3_CONFIG
from src.training.YOLOv8_trainer import YOLOv8Trainer
from src.evaluation.detection_evaluator import DetectionEvaluator


def main():
    """Run Experiment V3: YOLOv8 with Shallower Backbone."""
    
    print("=" * 80)
    print("EXPERIMENT V3: YOLOv8 with Shallower Backbone")
    print("=" * 80)
    
    # Configuration - Can use larger batch size due to smaller model
    DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
    EPOCHS = 80  # Fewer epochs for simpler model
    BATCH_SIZE = 20  # Increased batch size due to smaller model
    LR = 0.001
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'detection_yolov8_v3'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for this GPU: ≤20 (lighter model)')
    else:
        print('\nWarning: CUDA not available, using CPU (will be slow)')
    
    # Step 1: Load dataset config
    print("\n[1/5] Loading dataset configuration...")
    dataset_config_path = Path(DATASET_CONFIG)
    
    if not dataset_config_path.exists():
        print(f'Error: Dataset config not found: {dataset_config_path}')
        sys.exit(1)
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f'Dataset: {dataset_config["path"]}')
    print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
    
    # Step 2: Initialize model with custom YAML
    print("\n[2/5] Initializing YOLOv8 model with custom architecture...")
    model = YOLOv8Detector(**YOLOV8_V3_CONFIG)
    print(f'Model: Custom YOLOv8m (Shallower Backbone)')
    print(f'Input size: {YOLOV8_V3_CONFIG["input_size"]}')
    print(f'Custom YAML: {YOLOV8_V3_CONFIG["model_yaml"]}')
    print(f'Customization: Removed 8 convolutional layers from backbone')
    print(f'  - Reduced layer4 C2f: 6 → 3 repeats (removed 6 conv layers)')
    print(f'  - Reduced layer5 C2f: 3 → 2 repeats (removed 2 conv layers)')
    
    # Step 3: Train
    print("\n[3/5] Training model...")
    trainer = YOLOv8Trainer(
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        optimizer='adam',
        weight_decay=1e-4,
        use_amp=True
    )
    
    results = trainer.train(
        model=model,
        train_data=str(dataset_config_path),
        val_data=str(dataset_config_path),
        output_dir=str(output_dir / 'training'),
        patience=12  # Shorter patience for simpler model
    )
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating on test set...")
    evaluator = DetectionEvaluator()
    metrics = evaluator.evaluate_yolov8(
        model=model,
        test_dataset=str(dataset_config_path),
        output_dir=str(output_dir / 'evaluation')
    )
    
    # Step 5: Save summary and CSV metrics
    print("\n[5/5] Saving experiment summary and metrics...")
    
    # Save detailed metrics to CSV
    training_dir = output_dir / 'training' / 'train'
    if training_dir.exists():
        results_csv = training_dir / 'results.csv'
        if results_csv.exists():
            import shutil
            csv_output = output_dir / 'training' / 'training_history.csv'
            shutil.copy(results_csv, csv_output)
            print(f'Training history CSV saved to: {csv_output}')
    
    # Generate experiment summary
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment V3: YOLOv8 with Shallower Backbone\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Model: Custom YOLOv8m (Shallower Backbone)\n')
        f.write(f'- Input size: {YOLOV8_V3_CONFIG["input_size"]}\n')
        f.write(f'- Confidence threshold: {YOLOV8_V3_CONFIG["confidence_threshold"]}\n')
        f.write(f'- NMS IoU threshold: {YOLOV8_V3_CONFIG["nms_iou_threshold"]}\n')
        f.write(f'- Epochs: {EPOCHS}\n')
        f.write(f'- Batch size: {BATCH_SIZE}\n')
        f.write(f'- Learning rate: {LR}\n')
        f.write(f'- Optimizer: Adam\n')
        f.write(f'- Weight decay: 1e-4\n')
        f.write(f'- Mixed precision: Enabled\n')
        f.write(f'- Early stopping patience: 12\n\n')
        f.write(f'## Customization Details\n\n')
        f.write(f'- Type: **Removed Convolutional Layers**\n')
        f.write(f'- Location: Backbone layer4 and layer5\n')
        f.write(f'- Architecture changes:\n')
        f.write(f'  1. Reduced layer4 C2f module: 6 repeats → 3 repeats\n')
        f.write(f'     - Removed 3 bottleneck blocks (6 convolutional layers)\n')
        f.write(f'  2. Reduced layer5 C2f module: 3 repeats → 2 repeats\n')
        f.write(f'     - Removed 1 bottleneck block (2 convolutional layers)\n')
        f.write(f'- Total removed: **8 convolutional layers**\n')
        f.write(f'- Expected parameter decrease: ~12-15% (from ~25.9M to ~22-23M)\n')
        f.write(f'- Custom YAML: `{YOLOV8_V3_CONFIG["model_yaml"]}`\n\n')
        f.write(f'## Hypothesis\n\n')
        f.write(f'Reducing convolutional layers in the backbone should:\n')
        f.write(f'1. Decrease computational cost and memory usage\n')
        f.write(f'2. Enable larger batch sizes for more stable training\n')
        f.write(f'3. Potentially reduce overfitting on smaller datasets\n')
        f.write(f'4. Faster inference speed at the cost of some accuracy\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- mAP@0.5: {metrics["mAP50"]:.4f}\n')
        f.write(f'- mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}\n')
        f.write(f'- Precision: {metrics["precision"]:.4f}\n')
        f.write(f'- Recall: {metrics["recall"]:.4f}\n\n')
        f.write(f'## Analysis\n\n')
        f.write(f'This experiment tests whether a shallower backbone can maintain\n')
        f.write(f'reasonable performance while being more efficient.\n\n')
        f.write(f'Compared to V1 (baseline), we expect:\n')
        f.write(f'- Lower computational cost (fewer FLOPs)\n')
        f.write(f'- Faster training and inference\n')
        f.write(f'- Ability to use larger batch sizes\n')
        f.write(f'- Potential slight decrease in accuracy, especially for complex patterns\n\n')
        f.write(f'This lightweight variant may be suitable for deployment scenarios\n')
        f.write(f'where inference speed is prioritized over maximum accuracy.\n\n')
        f.write(f'## Files\n\n')
        f.write(f'- Training history: `training/training_history.csv`\n')
        f.write(f'- Best model: `training/train/weights/best.pt`\n')
        f.write(f'- Evaluation metrics: `evaluation/evaluation_metrics.json`\n')
        f.write(f'- Custom architecture: `src/models/yolov8m_custom_shallow.yaml`\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
