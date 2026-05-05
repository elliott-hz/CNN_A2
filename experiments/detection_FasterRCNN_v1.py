"""
Experiment V1: Faster R-CNN Baseline for Object Detection

Simple pipeline:
1. Load dataset
2. Initialize model (standard Faster R-CNN with ResNet50+FPN)
3. Train with CSV logging and validation
4. Evaluate
5. Save results and generate summary
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.FasterRCNNDetectorModel import FasterRCNNDetector, FASTERRCNN_V1_CONFIG
from src.training.FasterRCNN_trainer import FasterRCNNTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.data_processing.faster_rcnn_dataloader import create_faster_rcnn_dataloaders


def main():
    """Run Experiment V1: Faster R-CNN Baseline."""
    
    print("=" * 80)
    print("EXPERIMENT V1: Faster R-CNN Baseline")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Object_Detection/coco"  # Using COCO format
    ANNOTATION_FORMAT = 'coco'  # Options: 'coco', 'pascal', 'yolo'
    CLASS_NAMES = ['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']
    
    EPOCHS = 50
    BATCH_SIZE = 2  # Reduced for T4 GPU memory constraints (Faster R-CNN is memory-intensive)
    LR = 0.001
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'detection_fasterrcnn_v1'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for Faster R-CNN on this GPU: ≤2')
    else:
        print('\nWarning: CUDA not available, using CPU (will be very slow)')
    
    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    print(f'Data root: {DATA_ROOT}')
    print(f'Annotation format: {ANNOTATION_FORMAT}')
    print(f'Classes: {CLASS_NAMES}')
    
    try:
        train_loader, val_loader, test_loader = create_faster_rcnn_dataloaders(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
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
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing Faster R-CNN model...")
    
    # Update config with correct number of classes (including background)
    num_classes = len(CLASS_NAMES) + 1  # +1 for background
    model_config = {
        **FASTERRCNN_V1_CONFIG,
        'num_classes': num_classes
    }
    
    model = FasterRCNNDetector(**model_config)
    print(f'Number of classes: {num_classes} ({len(CLASS_NAMES)} + background)')
    print(f'Image size: {FASTERRCNN_V1_CONFIG["min_size"]}x{FASTERRCNN_V1_CONFIG["max_size"]}')
    print(f'Customization: None (Baseline)')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Step 3: Train
    print("\n[3/5] Training model...")
    trainer = FasterRCNNTrainer(learning_rate=LR, weight_decay=1e-4)
    
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        output_dir=str(output_dir / 'training'),
        patience=10
    )
    
    print(f'Training completed! Best val loss: {history["best_loss"]:.4f}')
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    evaluator = DetectionEvaluator()
    
    # Note: Faster R-CNN evaluation needs proper implementation
    # For now, we'll use training loss as proxy metric
    print('Note: Detailed mAP evaluation for Faster R-CNN requires additional implementation.')
    print('Using training loss as proxy metric.')
    
    metrics = {
        'best_val_loss': float(history['best_loss']),
        'note': 'Full mAP evaluation pending implementation'
    }
    
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
    customization_desc = "None (Baseline - standard Faster R-CNN with ResNet50+FPN)"
    
    evaluator.generate_experiment_summary(
        output_dir=str(output_dir),
        experiment_name="V1: Faster R-CNN Baseline",
        model_config=model_config,
        training_config={
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'weight_decay': 1e-4,
            'patience': 10
        },
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
