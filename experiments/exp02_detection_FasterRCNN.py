"""
Experiment 02: Faster R-CNN Baseline for Dog Detection

Simple pipeline:
1. Load dataset
2. Initialize model
3. Train
4. Evaluate
5. Save results
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.FasterRCNNDetectorModel import FasterRCNNDetector, FASTERRCNN_BASELINE_CONFIG
from src.training.FasterRCNN_trainer import FasterRCNNTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.data_processing.faster_rcnn_dataloader import create_faster_rcnn_dataloaders


def main():
    """Run Experiment 02: Faster R-CNN Baseline."""
    
    print("=" * 80)
    print("EXPERIMENT 02: Faster R-CNN Baseline")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Object_Detection/coco"  # Using COCO format
    ANNOTATION_FORMAT = 'coco'  # Options: 'coco', 'pascal', 'yolo'
    CLASS_NAMES = ['Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified']
    
    EPOCHS = 50
    BATCH_SIZE = 2  # Reduced from 4 for T4 GPU memory constraints (Faster R-CNN is memory-intensive)
    LR = 0.001
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'exp02_fasterrcnn'
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
        **FASTERRCNN_BASELINE_CONFIG,
        'num_classes': num_classes
    }
    
    model = FasterRCNNDetector(**model_config)
    print(f'Number of classes: {num_classes} ({len(CLASS_NAMES)} + background)')
    print(f'Image size: {FASTERRCNN_BASELINE_CONFIG["min_size"]}x{FASTERRCNN_BASELINE_CONFIG["max_size"]}')
    
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
    
    print(f'Training completed! Best loss: {history["best_loss"]:.4f}')
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    evaluator = DetectionEvaluator()
    
    # Note: Faster R-CNN evaluation needs proper implementation
    # For now, we'll skip detailed metrics
    print('Note: Detailed mAP evaluation for Faster R-CNN requires additional implementation.')
    print('Using training loss as proxy metric.')
    
    metrics = {
        'best_training_loss': float(history['best_loss']),
        'note': 'Full mAP evaluation pending implementation'
    }
    
    # Save metrics
    metrics_path = output_dir / 'evaluation' / 'evaluation_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Step 5: Save summary
    print("\n[5/5] Saving experiment summary...")
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment 02: Faster R-CNN Baseline\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Model: Faster R-CNN with ResNet50+FPN\n')
        f.write(f'- Number of classes: {num_classes} ({len(CLASS_NAMES)} + background)\n')
        f.write(f'- Class names: {", ".join(CLASS_NAMES)}\n')
        f.write(f'- Image size: {FASTERRCNN_BASELINE_CONFIG["min_size"]}\n')
        f.write(f'- Dataset: {DATA_ROOT}\n')
        f.write(f'- Annotation format: {ANNOTATION_FORMAT}\n')
        f.write(f'- Epochs: {EPOCHS}\n')
        f.write(f'- Batch size: {BATCH_SIZE}\n')
        f.write(f'- Learning rate: {LR}\n')
        f.write(f'- Weight decay: 1e-4\n\n')
        f.write(f'## Dataset Statistics\n\n')
        f.write(f'- Train samples: {len(train_loader.dataset)}\n')
        f.write(f'- Val samples: {len(val_loader.dataset)}\n')
        f.write(f'- Test samples: {len(test_loader.dataset)}\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- Best Training Loss: {history["best_loss"]:.4f}\n')
        f.write(f'- Training completed: Yes\n')
        f.write(f'- Full mAP evaluation: Pending implementation\n\n')
        f.write(f'## Notes\n\n')
        f.write(f'- Dataloader supports COCO, Pascal VOC, and YOLO formats\n')
        f.write(f'- Training uses custom loop with Adam optimizer\n')
        f.write(f'- Early stopping enabled (patience=10)\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
