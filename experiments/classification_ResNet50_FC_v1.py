"""
Experiment: ResNet50 FC v1 Classification

Enhanced multi-layer FC head with stronger regularization.
All layers trainable (NO freezing) - following teacher's methodology requirements.

Usage:
    python experiments/classification_ResNet50_FC_v1.py [--pretrained True/False] [--dataAugmentation none/standard/enhanced]
    
    --pretrained: Use pretrained ImageNet weights (default: True from config)
                  Set to False to train from scratch
    --dataAugmentation: Data augmentation strategy (default: 'none')
                        Options: 'none', 'standard', 'enhanced'
"""

import sys
from pathlib import Path
import torch
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, CUSTOMIZED_FC_V1_CONFIG
from src.training.ResNet50_trainer import ClassificationTrainer, TRAINING_CONFIG_FC_V1
from src.data_processing.ClassificationDataLoader import create_classification_dataloaders
from src.evaluation.classification_evaluator import ClassificationEvaluator


def main():
    """Run Customized FC v1 Experiment."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ResNet50 FC v1 Classification Experiment')
    parser.add_argument('--pretrained', type=str, default=None, 
                       help='Use pretrained weights: True, False, or None (use config default)')
    parser.add_argument('--dataAugmentation', type=str, default='none',
                       choices=['none', 'standard', 'enhanced'],
                       help='Data augmentation strategy (default: none)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENT: ResNet50 FC v1")
    print("=" * 80)
    
    # Configuration
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'FC_v1'
    output_dir = Path(f'outputs/classification_{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
    else:
        print('\nWarning: CUDA not available, using CPU')
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, test_loader, class_names = create_classification_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, augmentation_type=args.dataAugmentation
    )
    print(f'Classes: {class_names}')
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    # Print augmentation info
    aug_descriptions = {
        'none': 'No augmentation (basic preprocessing only)',
        'standard': 'Standard (Rotation 15°, ColorJitter 0.2)',
        'enhanced': 'Enhanced (Rotation 20°, ColorJitter 0.3+hue, RandomAffine)'
    }
    print(f'Data augmentation: {aug_descriptions[args.dataAugmentation]}')
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing customized model...")
    
    # Handle pretrained parameter
    model_config = CUSTOMIZED_FC_V1_CONFIG.copy()
    if args.pretrained is not None:
        model_config['pretrained'] = args.pretrained.lower() == 'true'
    
    print('Modifications:')
    print('  - Enhanced multi-layer FC head (2048 → 512 → 256 → 10) with BatchNorm')
    print('  - Higher dropout (0.7)')
    print('  - ALL layers trainable (NO freezing)')
    
    if model_config['pretrained']:
        print('Pretrained: YES (ImageNet weights)')
    else:
        print('Pretrained: NO (Training from scratch)')
        print('Note: Training configuration includes warmup and extended training schedule')
    
    model = ResNet50Classifier(**model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')
    
    # Print detailed model architecture
    print("\nModel Summary:")
    trainer_temp = ClassificationTrainer(model, config=TRAINING_CONFIG_FC_V1)
    trainer_temp.print_model_summary()
    
    # Step 3: Train
    print("\n[3/5] Training...")
    trainer = trainer_temp  # Reuse the trainer we created for model summary
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_FC_V1.label_smoothing)
    
    history = trainer.train(
        train_loader, val_loader, criterion,
        str(output_dir / 'training')
    )
    print(f'Best Val Acc: {trainer.best_val_acc:.4f}')
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = output_dir / 'training' / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(trainer.device)
        print(f'✓ Loaded best model from epoch {checkpoint["epoch"]} (Val Acc: {checkpoint["val_acc"]:.4f})')
    else:
        print('Warning: Best model checkpoint not found, using final model')
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating on test set...")
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.evaluate(model, test_loader, str(output_dir / 'evaluation'))
    
    # Step 5: Generate visualization and summary
    print("\n[5/5] Generating training curves, analysis, and summary...")
    evaluator.plot_training_curves(history['history'], str(output_dir / 'visualization'))
    
    analysis = evaluator.analyze_overfitting(history['history'])
    
    # Generate comprehensive summary
    evaluator.generate_experiment_summary(
        experiment_name=experiment_name,
        model_config=model_config,
        training_config=TRAINING_CONFIG_FC_V1,
        trainer_metrics={'best_val_acc': trainer.best_val_acc},
        evaluation_metrics=metrics,
        overfitting_analysis=analysis,
        output_dir=str(output_dir)
    )
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
