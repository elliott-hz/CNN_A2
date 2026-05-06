"""
Experiment V3: YOLOv8 with Shallower Backbone (Reduced Convolutional Layers)

This experiment removes convolutional layers from the YOLOv8 backbone:
- Reduces C2f module repeats in layer4 from 6 to 3
- Removes 3 bottlenecks (6 convolutional layers total)
- Tests if a lighter model can maintain performance with faster inference

Customization Details:
- Location: Backbone layer4 (P4/16 level)
- Removed layers:
  * Reduced layer4 C2f from 6 to 3 repeats (removed 3 bottlenecks = 6 conv layers)
- Total removed: 6 convolutional layers
- Expected parameter decrease: ~3M (~22.9M total)
- All layer indices remain unchanged (stable architecture)
"""

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_V3_CONFIG as MODEL_V3_CONFIG
from src.training.YOLOv8_trainer import YOLOv8Trainer, YOLOV8_V3_CONFIG as TRAIN_V3_CONFIG
from src.evaluation.detection_evaluator import DetectionEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run YOLOv8 V3 Shallower Backbone Experiment')
    parser.add_argument('--pretrained', type=str, default='True', 
                        choices=['True', 'False'],
                        help='Whether to use pretrained weights (default: True)')
    return parser.parse_args()


def main():
    """Run Experiment V3: YOLOv8 with Shallower Backbone."""
    args = parse_args()
    use_pretrained = args.pretrained.lower() == 'true'
    
    print("=" * 80)
    print("EXPERIMENT V3: YOLOv8 with Shallower Backbone")
    print("=" * 80)
    print(f"Use Pretrained Weights: {use_pretrained}")
    
    # Configuration
    DATASET_CONFIG = "data/25509225/Object_Detection/yolo/data.yaml"
    
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
    
    print(f'Dataset config: {dataset_config_path}')
    print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
    print(f'Train path: {dataset_config.get("train", "N/A")}')
    print(f'Val path: {dataset_config.get("val", "N/A")}')
    print(f'Test path: {dataset_config.get("test", "N/A")}')
    
    # Step 2: Initialize model with custom architecture
    print("\n[2/5] Initializing YOLOv8 model with custom architecture...")
    # Update model config with the command line argument
    model_config = MODEL_V3_CONFIG.copy()
    model_config['pretrained'] = use_pretrained
    
    model = YOLOv8Detector(**model_config)
    print(f'Model: Custom YOLOv8m (Shallower Backbone)')
    print(f'Input size: {MODEL_V3_CONFIG["input_size"]}')
    print(f'Pretrained: {use_pretrained}')
    print(f'Customization type: {MODEL_V3_CONFIG["customize_type"]}')
    print(f'Customization: Reduced backbone depth by modifying C2f modules')
    print(f'  - Location: Layer4 (P4/16 level)')
    print(f'  - Change: Reduced C2f repeats (lighter model)')
    print(f'  - Benefit: Faster training, reduced overfitting risk')
    
    # Step 3: Train using centralized configuration
    print("\n[3/5] Training model...")
    trainer = YOLOv8Trainer(**TRAIN_V3_CONFIG)
    
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
    
    # Step 5: Save summary and CSV metrics + Visualizations
    print("\n[5/5] Saving experiment summary, metrics, and visualizations...")
    
    # Organize training artifacts (CSV + plots) using evaluator
    evaluator.organize_training_artifacts(str(output_dir))
    
    # Generate experiment summary using evaluator
    customization_desc = (
        "Reduced backbone depth using PyTorch-based modification:\n"
        "- Modified C2f modules to use reduced repetition counts\n"
        "- Reduces number of convolutional layers in deeper parts of backbone\n"
        "- Maintains stable architecture indices (no structural changes)\n"
        "- Benefits: Faster inference, lower memory, reduced overfitting\n"
        "- Purpose: Test if lighter model maintains reasonable performance"
    )
    
    evaluator.generate_experiment_summary(
        output_dir=str(output_dir),
        experiment_name="V3: YOLOv8 Shallower Backbone",
        model_config=model_config,
        training_config=TRAIN_V3_CONFIG,
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
