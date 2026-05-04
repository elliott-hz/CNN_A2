"""
Experiment V2: YOLOv8 with Deeper Backbone (Added Convolutional Layers)

This experiment adds extra convolutional layers to the YOLOv8 backbone:
- Inserts additional C2f module between layer3 and layer4
- Adds 6 new convolutional layers total
- Tests if deeper backbone improves feature extraction

Customization Details:
- Location: After backbone layer3 (before layer4)
- Added layers: 
  * 1x1 Conv (1024→512 channels) - dimensionality reduction
  * C2f module with 2 bottlenecks (4 conv layers) - feature enhancement
  * 3x3 Conv (512→1024 channels) - dimensionality restoration
- Total added: 6 convolutional layers
- Expected parameter increase: ~15-18%
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
import yaml
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_V2_CONFIG
from src.training.YOLOv8_trainer import YOLOv8Trainer
from src.evaluation.detection_evaluator import DetectionEvaluator


def main():
    """Run Experiment V2: YOLOv8 with Deeper Backbone."""
    
    print("=" * 80)
    print("EXPERIMENT V2: YOLOv8 with Deeper Backbone")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU with larger model
    DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
    EPOCHS = 120  # More epochs for deeper model
    BATCH_SIZE = 12  # Reduced batch size due to larger model
    LR = 0.0005  # Lower learning rate for stability
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'detection_yolov8_v2'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for this GPU: ≤12 (deeper model)')
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
    model = YOLOv8Detector(**YOLOV8_V2_CONFIG)
    print(f'Model: Custom YOLOv8m (Deeper Backbone)')
    print(f'Input size: {YOLOV8_V2_CONFIG["input_size"]}')
    print(f'Custom YAML: {YOLOV8_V2_CONFIG["model_yaml"]}')
    print(f'Customization: Added 6 convolutional layers in backbone')
    print(f'  - 1x1 Conv (dimensionality reduction)')
    print(f'  - C2f module with 2 bottlenecks (4 conv layers)')
    print(f'  - 3x3 Conv (dimensionality restoration)')
    
    # Step 3: Train
    print("\n[3/5] Training model...")
    trainer = YOLOv8Trainer(
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        optimizer='adam',
        weight_decay=5e-4,  # Stronger regularization for deeper model
        use_amp=True
    )
    
    results = trainer.train(
        model=model,
        train_data=str(dataset_config_path),
        val_data=str(dataset_config_path),
        output_dir=str(output_dir / 'training'),
        patience=20,  # Longer patience for deeper model
        cos_lr=True   # Use cosine learning rate schedule
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
        f.write(f'# Experiment V2: YOLOv8 with Deeper Backbone\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Model: Custom YOLOv8m (Deeper Backbone)\n')
        f.write(f'- Input size: {YOLOV8_V2_CONFIG["input_size"]}\n')
        f.write(f'- Confidence threshold: {YOLOV8_V2_CONFIG["confidence_threshold"]}\n')
        f.write(f'- NMS IoU threshold: {YOLOV8_V2_CONFIG["nms_iou_threshold"]}\n')
        f.write(f'- Epochs: {EPOCHS}\n')
        f.write(f'- Batch size: {BATCH_SIZE}\n')
        f.write(f'- Learning rate: {LR}\n')
        f.write(f'- Optimizer: Adam\n')
        f.write(f'- Weight decay: 5e-4 (stronger regularization)\n')
        f.write(f'- Mixed precision: Enabled\n')
        f.write(f'- Early stopping patience: 20\n')
        f.write(f'- Learning rate schedule: Cosine annealing\n\n')
        f.write(f'## Customization Details\n\n')
        f.write(f'- Type: **Added Convolutional Layers**\n')
        f.write(f'- Location: Backbone (between layer3 and layer4)\n')
        f.write(f'- Architecture changes:\n')
        f.write(f'  1. Added 1×1 Conv layer (1024→512 channels) for dimensionality reduction\n')
        f.write(f'  2. Added C2f module with 2 bottleneck blocks (4 convolutional layers)\n')
        f.write(f'  3. Added 3×3 Conv layer (512→1024 channels) for dimensionality restoration\n')
        f.write(f'- Total added: **6 convolutional layers**\n')
        f.write(f'- Expected parameter increase: ~15-18% (from ~25.9M to ~29-30M)\n')
        f.write(f'- Custom YAML: `{YOLOV8_V2_CONFIG["model_yaml"]}`\n\n')
        f.write(f'## Hypothesis\n\n')
        f.write(f'Adding convolutional layers in the backbone should:\n')
        f.write(f'1. Enhance feature extraction capability\n')
        f.write(f'2. Capture more complex patterns in solar panel damage\n')
        f.write(f'3. Potentially improve detection accuracy for subtle defects\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- mAP@0.5: {metrics["mAP50"]:.4f}\n')
        f.write(f'- mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}\n')
        f.write(f'- Precision: {metrics["precision"]:.4f}\n')
        f.write(f'- Recall: {metrics["recall"]:.4f}\n\n')
        f.write(f'## Analysis\n\n')
        f.write(f'This experiment tests whether a deeper backbone improves detection performance.\n')
        f.write(f'The additional convolutional layers provide more capacity for feature learning,\n')
        f.write(f'but may also increase the risk of overfitting if not properly regularized.\n\n')
        f.write(f'Compared to V1 (baseline), we expect:\n')
        f.write(f'- Higher computational cost (more FLOPs)\n')
        f.write(f'- Potentially better accuracy if the dataset benefits from deeper features\n')
        f.write(f'- May require more training epochs to converge\n\n')
        f.write(f'## Files\n\n')
        f.write(f'- Training history: `training/training_history.csv`\n')
        f.write(f'- Best model: `training/train/weights/best.pt`\n')
        f.write(f'- Evaluation metrics: `evaluation/evaluation_metrics.json`\n')
        f.write(f'- Custom architecture: `src/models/yolov8m_custom_deeper.yaml`\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
