"""
Experiment: ResNet50 Customized v1 Classification

Enhanced multi-layer FC head with stronger regularization.
All layers trainable (NO freezing) - following teacher's methodology requirements.
"""

import sys
from pathlib import Path
import torch
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, CUSTOMIZED_V1_CONFIG
from src.training.classification_trainer import ClassificationTrainer, TRAINING_CONFIG_V1
from src.data_processing.ClassificationDataLoader import create_enhanced_dataloaders
from src.evaluation.classification_evaluator import ClassificationEvaluator


def main():
    """Run Customized v1 Experiment."""
    
    print("=" * 80)
    print("EXPERIMENT: ResNet50 Customized v1")
    print("=" * 80)
    
    # Configuration
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'customized_v1'
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
    train_loader, val_loader, test_loader, class_names = create_enhanced_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE
    )
    print(f'Classes: {class_names}')
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    print('Data augmentation: Enhanced (Rotation 20°, ColorJitter 0.3+hue, RandomAffine)')
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing customized model...")
    print('Modifications:')
    print('  - Enhanced multi-layer FC head (2048 → 512 → 256 → 10) with BatchNorm')
    print('  - Higher dropout (0.7)')
    print('  - Enhanced data augmentation')
    print('  - ALL layers trainable (NO freezing)')
    
    model = ResNet50Classifier(**CUSTOMIZED_V1_CONFIG)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')
    
    # Step 3: Train
    print("\n[3/5] Training...")
    trainer = ClassificationTrainer(model, config=TRAINING_CONFIG_V1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG_V1.label_smoothing)
    
    history = trainer.train(
        train_loader, val_loader, criterion,
        str(output_dir / 'training')
    )
    print(f'Best Val Acc: {trainer.best_val_acc:.4f}')
    
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
        model_config=CUSTOMIZED_V1_CONFIG,
        training_config=TRAINING_CONFIG_V1,
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
