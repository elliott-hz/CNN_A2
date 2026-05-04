"""
Experiment V1: YOLOv8 Baseline for Object Detection

Simple pipeline:
1. Load dataset config
2. Initialize model (standard YOLOv8m)
3. Train
4. Evaluate
5. Save results and generate CSV metrics log
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
import yaml
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_BASELINE_CONFIG
from src.training.YOLOv8_trainer import YOLOv8Trainer
from src.evaluation.detection_evaluator import DetectionEvaluator


def main():
    """Run Experiment 01: YOLOv8 Baseline."""
    
    print("=" * 80)
    print("EXPERIMENT 01: YOLOv8 Baseline")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
    EPOCHS = 100
    BATCH_SIZE = 16  # Optimized for T4 GPU
    LR = 0.001
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'detection_yolov8_v1'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for this GPU: ≤16')
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
    
    print(f'Dataset config: {dataset_config_path}')
    print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
    print(f'Train path: {dataset_config.get("train", "N/A")}')
    print(f'Val path: {dataset_config.get("val", "N/A")}')
    print(f'Test path: {dataset_config.get("test", "N/A")}')
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing YOLOv8 model...")
    model = YOLOv8Detector(**YOLOV8_BASELINE_CONFIG)
    print(f'Model: Standard YOLOv8{YOLOV8_BASELINE_CONFIG["backbone"]}')
    print(f'Input size: {YOLOV8_BASELINE_CONFIG["input_size"]}')
    print(f'Customization: None (Baseline)')
    
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
        patience=15  # Early stopping
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
            # Copy to standardized location
            import shutil
            csv_output = output_dir / 'training' / 'training_history.csv'
            shutil.copy(results_csv, csv_output)
            print(f'Training history CSV saved to: {csv_output}')
    
    # Generate experiment summary
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment V1: YOLOv8 Baseline\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Model: Standard YOLOv8{YOLOV8_BASELINE_CONFIG["backbone"]}\n')
        f.write(f'- Input size: {YOLOV8_BASELINE_CONFIG["input_size"]}\n')
        f.write(f'- Confidence threshold: {YOLOV8_BASELINE_CONFIG["confidence_threshold"]}\n')
        f.write(f'- NMS IoU threshold: {YOLOV8_BASELINE_CONFIG["nms_iou_threshold"]}\n')
        f.write(f'- Epochs: {EPOCHS}\n')
        f.write(f'- Batch size: {BATCH_SIZE}\n')
        f.write(f'- Learning rate: {LR}\n')
        f.write(f'- Optimizer: Adam\n')
        f.write(f'- Weight decay: 1e-4\n')
        f.write(f'- Mixed precision: Enabled\n')
        f.write(f'- Early stopping patience: 15\n\n')
        f.write(f'## Customization\n\n')
        f.write(f'- Type: None (Baseline)\n')
        f.write(f'- Description: Standard YOLOv8m architecture without modifications\n')
        f.write(f'- Purpose: Control group for comparison with customized variants\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- mAP@0.5: {metrics["mAP50"]:.4f}\n')
        f.write(f'- mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}\n')
        f.write(f'- Precision: {metrics["precision"]:.4f}\n')
        f.write(f'- Recall: {metrics["recall"]:.4f}\n\n')
        f.write(f'## Analysis\n\n')
        f.write(f'This baseline experiment uses the standard YOLOv8m architecture.\n')
        f.write(f'It serves as a reference point for evaluating the impact of architectural\n')
        f.write(f'modifications in V2 (deeper backbone) and V3 (shallower backbone).\n\n')
        f.write(f'## Files\n\n')
        f.write(f'- Training history: `training/training_history.csv`\n')
        f.write(f'- Best model: `training/train/weights/best.pt`\n')
        f.write(f'- Evaluation metrics: `evaluation/evaluation_metrics.json`\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
