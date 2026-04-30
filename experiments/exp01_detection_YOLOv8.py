"""
Experiment 01: YOLOv8 Baseline for Dog Detection

Simple pipeline:
1. Load dataset config
2. Initialize model
3. Train
4. Evaluate
5. Save results
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

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
    
    # Configuration
    DATASET_CONFIG = "data/processed/detection/dataset.yaml"
    EPOCHS = 100
    BATCH_SIZE = 24
    LR = 0.001
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/exp01_yolov8_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset config
    print("\n[1/5] Loading dataset configuration...")
    dataset_config_path = Path(DATASET_CONFIG)
    
    if not dataset_config_path.exists():
        print(f'Error: Dataset config not found: {dataset_config_path}')
        print('Please run preprocessing first.')
        sys.exit(1)
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f'Dataset: {dataset_config["path"]}')
    print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing YOLOv8 model...")
    model = YOLOv8Detector(**YOLOV8_BASELINE_CONFIG)
    print(f'Backbone: {YOLOV8_BASELINE_CONFIG["backbone"]}')
    print(f'Input size: {YOLOV8_BASELINE_CONFIG["input_size"]}')
    
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
    
    # Step 5: Save summary
    print("\n[5/5] Saving experiment summary...")
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment 01: YOLOv8 Baseline\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Model: YOLOv8{YOLOV8_BASELINE_CONFIG["backbone"]}\n')
        f.write(f'- Input size: {YOLOV8_BASELINE_CONFIG["input_size"]}\n')
        f.write(f'- Confidence threshold: {YOLOV8_BASELINE_CONFIG["confidence_threshold"]}\n')
        f.write(f'- Epochs: {EPOCHS}\n')
        f.write(f'- Batch size: {BATCH_SIZE}\n')
        f.write(f'- Learning rate: {LR}\n')
        f.write(f'- Optimizer: Adam\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- mAP@0.5: {metrics["mAP50"]:.4f}\n')
        f.write(f'- mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}\n')
        f.write(f'- Precision: {metrics["precision"]:.4f}\n')
        f.write(f'- Recall: {metrics["recall"]:.4f}\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
