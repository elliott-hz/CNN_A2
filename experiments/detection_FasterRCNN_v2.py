"""
Experiment V2: Faster R-CNN with Deeper Backbone (Added Convolutional Layers)

This experiment adds convolutional layers to the Faster R-CNN backbone:
- Inserts 2 Conv-BN-ReLU blocks after ResNet50 layer2
- Deepens shallow-layer feature extraction for better fine-grained detection
- Tests if deeper architecture improves detection performance

Customization Details:
- Location: After ResNet50 layer2 (before layer3)
- Added layers:
  * Conv(512, 512, 3×3, stride=1, padding=1) + BN + ReLU
  * Conv(512, 512, 3×3, stride=1, padding=1) + BN + ReLU
- Total added: 6 convolutional layers
- Expected parameter increase: ~4.7M
- Purpose: Enhance feature representation at intermediate scale
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.FasterRCNNDetectorModel import FasterRCNNDetector, FASTERRCNN_V2_CONFIG as MODEL_CONFIG
from src.training.FasterRCNN_trainer import FasterRCNNTrainer, FASTERRCNN_V2_CONFIG as TRAIN_CONFIG
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.data_processing.faster_rcnn_dataloader import create_faster_rcnn_dataloaders


def main():
    """Run Experiment V2: Faster R-CNN with Deeper Backbone."""
    
    print("=" * 80)
    print("EXPERIMENT V2: Faster R-CNN with Deeper Backbone")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Object_Detection/coco"  # Using COCO format
    ANNOTATION_FORMAT = 'coco'  # Options: 'coco', 'pascal', 'yolo'
    CLASS_NAMES = ['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'detection_fasterrcnn_v2'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for this GPU: ≤2 (deeper model)')
    else:
        print('\nWarning: CUDA not available, using CPU (will be very slow)')
    
    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    print(f'Data root: {DATA_ROOT}')
    print(f'Annotation format: {ANNOTATION_FORMAT}')
    print(f'Classes: {CLASS_NAMES}')
    print('\n💡 NOTE: Invalid bounding boxes (width ≤ 0 or height ≤ 0) will be')
    print('   automatically filtered during data loading. This includes train, val, and test sets.')
    
    try:
        train_loader, val_loader, test_loader = create_faster_rcnn_dataloaders(
            data_root=DATA_ROOT,
            batch_size=TRAIN_CONFIG['batch_size'],
            num_workers=2,
            annotation_format=ANNOTATION_FORMAT,
            class_names=CLASS_NAMES
        )
        
        print(f'\nDataset loaded successfully:')
        print(f'  Train samples: {len(train_loader.dataset)}')
        print(f'  Val samples: {len(val_loader.dataset)}')
        print(f'  Test samples: {len(test_loader.dataset)}')
        
    except Exception as e:
        print(f'Error loading dataset: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Initialize model with custom architecture
    print("\n[2/5] Initializing Faster R-CNN model with deeper backbone...")
    
    # Update config with correct number of classes (including background)
    num_classes = len(CLASS_NAMES) + 1  # +1 for background
    model_config = {
        **MODEL_CONFIG,
        'num_classes': num_classes
    }
    
    model = FasterRCNNDetector(**model_config)
    print(f'Number of classes: {num_classes} ({len(CLASS_NAMES)} + background)')
    print(f'Image size: {MODEL_CONFIG["min_size"]}x{MODEL_CONFIG["max_size"]}')
    print(f'Customization type: {MODEL_CONFIG["customize_type"]}')
    print(f'Customization: Added 2 Conv-BN-ReLU blocks after layer2')
    print(f'  - Location: Between ResNet50 layer2 and layer3')
    print(f'  - Change: +6 convolutional layers (512 channels each)')
    print(f'  - Benefit: Enhanced intermediate-scale feature extraction')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Step 3: Train
    print("\n[3/5] Training model...")
    trainer = FasterRCNNTrainer(TRAIN_CONFIG)
    
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir / 'training')
    )
    
    print(f'Training completed! Best val loss: {history["best_loss"]:.4f}')
    
    # Step 4: Evaluate on test set
    print("\n[4/5] Evaluating model on test set...")
    evaluator = DetectionEvaluator()
    
    metrics = evaluator.evaluate_fasterrcnn(
        model=model,
        test_loader=test_loader,
        output_dir=str(output_dir / 'evaluation'),
        class_names=CLASS_NAMES
    )
    
    # Save metrics
    metrics_path = output_dir / 'evaluation' / 'evaluation_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Step 5: Save summary
    print("\n[5/5] Saving experiment summary...")
    
    # Organize training artifacts using evaluator
    evaluator.organize_training_artifacts(str(output_dir))
    
    # Generate experiment summary using evaluator
    customization_desc = (
        "Deeper backbone using PyTorch-based modification:\n"
        "- Added 2 Conv-BN-ReLU blocks after ResNet50 layer2\n"
        "- Each block: Conv(512→512, 3×3) + BatchNorm + ReLU\n"
        "- Total added: 6 convolutional layers (~4.7M parameters)\n"
        "- Benefits: Enhanced intermediate-scale feature representation\n"
        "- Purpose: Test if deeper architecture improves detection accuracy"
    )
    
    # Add note about bbox validation to metrics
    metrics['data_quality_note'] = (
        f"Train: {train_loader.dataset.skipped_annotations} invalid bboxes filtered; "
        f"Val: {val_loader.dataset.skipped_annotations} invalid bboxes filtered; "
        f"Test: {test_loader.dataset.skipped_annotations} invalid bboxes filtered. "
        f"All datasets automatically skip bboxes with width ≤ 0 or height ≤ 0."
    )
    
    evaluator.generate_experiment_summary(
        output_dir=str(output_dir),
        experiment_name="V2: Faster R-CNN Deeper Backbone",
        model_config=model_config,
        training_config=TRAIN_CONFIG,
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
