"""
Experiment V2: YOLOv8 with Deeper Backbone (Added Convolutional Layers)

This experiment adds extra convolutional layers to the YOLOv8 backbone:
- Inserts 2 Conv layers after layer2 (C2f module)
- Adds 2 new convolutional layers total
- Tests if deeper backbone improves feature extraction in shallow layers

Customization Details:
- Location: After backbone layer2 (C2f with 128 channels)
- Added layers: 
  * Conv(128, 3x3, stride=1, padding=1) - maintain resolution
  * Conv(128, 3x3, stride=1, padding=1) - deepen feature representation
- Total added: 2 convolutional layers
- Expected parameter increase: ~0.5M (~26.4M total)
- All subsequent layer indices shifted by +2
"""

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_V2_CONFIG as MODEL_V2_CONFIG
from src.training.YOLOv8_trainer import YOLOv8Trainer, YOLOV8_V2_CONFIG as TRAIN_V2_CONFIG
from src.evaluation.detection_evaluator import DetectionEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run YOLOv8 V2 Deeper Backbone Experiment')
    parser.add_argument('--pretrained', type=str, default='True', 
                        choices=['True', 'False'],
                        help='Whether to use pretrained weights (default: True)')
    return parser.parse_args()


def main():
    """Run Experiment V2: YOLOv8 with Deeper Backbone."""
    args = parse_args()
    use_pretrained = args.pretrained.lower() == 'true'
    
    print("=" * 80)
    print("EXPERIMENT V2: YOLOv8 with Deeper Backbone")
    print("=" * 80)
    print(f"Use Pretrained Weights: {use_pretrained}")
    
    # Configuration
    DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
    
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
    
    print(f'Dataset config: {dataset_config_path}')
    print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
    print(f'Train path: {dataset_config.get("train", "N/A")}')
    print(f'Val path: {dataset_config.get("val", "N/A")}')
    print(f'Test path: {dataset_config.get("test", "N/A")}')
    
    # Step 2: Initialize model with custom YAML
    print("\n[2/5] Initializing YOLOv8 model with custom architecture...")
    # Update model config with the command line argument
    model_config = MODEL_V2_CONFIG.copy()
    model_config['pretrained'] = use_pretrained
    
    model = YOLOv8Detector(**model_config)
    print(f'Model: Custom YOLOv8m (Deeper Backbone)')
    print(f'Input size: {MODEL_V2_CONFIG["input_size"]}')
    print(f'Pretrained: {use_pretrained}')
    print(f'Custom YAML: {MODEL_V2_CONFIG["model_yaml"]}')
    print(f'Customization: Added 2 convolutional layers in backbone')
    print(f'  - Location: After layer2 (C2f module)')
    print(f'  - Layer 3: Conv(128, 3x3, stride=1, padding=1)')
    print(f'  - Layer 4: Conv(128, 3x3, stride=1, padding=1)')
    print(f'  - Purpose: Deepen shallow-layer feature extraction')
    
    # Step 3: Train using centralized configuration
    print("\n[3/5] Training model...")
    trainer = YOLOv8Trainer(**TRAIN_V2_CONFIG)
    
    results = trainer.train(
        model=model,
        train_data=str(dataset_config_path),
        val_data=str(dataset_config_path),
        output_dir=str(output_dir / 'training')
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
    
    # Generate experiment summary using evaluator
    customization_desc = (
        "Added 2 convolutional layers in Backbone after layer2:\n"
        "- Inserted Conv(128, 3x3, s=1, p=1) x2 after C2f module\n"
        "- Maintains spatial resolution while deepening feature extraction\n"
        "- Increases model depth for better shallow feature learning\n"
        "- Parameter increase: ~0.5M"
    )
    
    evaluator.generate_experiment_summary(
        output_dir=str(output_dir),
        experiment_name="V2: YOLOv8 Deeper Backbone",
        model_config=model_config,
        training_config=TRAIN_V2_CONFIG,
        metrics=metrics,
        customization_details=customization_desc
    )
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {output_dir / "experiment_summary.md"}')


if __name__ == '__main__':
    main()
