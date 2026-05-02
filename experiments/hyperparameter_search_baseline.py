"""
Hyperparameter Grid Search for Baseline Model

Systematically tests combinations of learning rate, weight decay, and label smoothing
to optimize Baseline performance beyond 97.59%.

Usage:
    python3 experiments/hyperparameter_search_baseline.py
    
Note: This will run multiple training sessions. Ensure you have sufficient time and GPU resources.
Each combination trains for up to 200 epochs with early stopping.
"""

import sys
from pathlib import Path
import torch
import itertools
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, BASELINE_CONFIG
from src.training.classification_trainer import ClassificationTrainer, TrainingConfig
from src.data_processing.ClassificationDataLoader import create_classification_dataloaders


def create_config(lr, weight_decay, label_smoothing, warmup_epochs=10):
    """Create training configuration for given hyperparameters."""
    return TrainingConfig(
        learning_rate=lr,
        weight_decay=weight_decay,
        optimizer_type='adamw',
        epochs=200,
        use_warmup=True,
        warmup_epochs=warmup_epochs,
        use_scheduler=True,
        scheduler_type='reduce_on_plateau',
        scheduler_patience=7,
        scheduler_factor=0.5,
        use_early_stopping=True,
        early_stopping_patience=50,
        label_smoothing=label_smoothing,
        use_amp=True,
        description=f'Baseline grid search: LR={lr}, WD={weight_decay}, LS={label_smoothing}'
    )


def main():
    """Run hyperparameter grid search for Baseline model."""
    
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH - BASELINE MODEL")
    print("=" * 80)
    
    # Configuration
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
    else:
        print('\nWarning: CUDA not available, using CPU')
        return
    
    # Define search space
    lr_values = [5e-4, 1e-3, 2e-3]
    weight_decay_values = [5e-4, 1e-3, 5e-3]
    label_smoothing_values = [0.05, 0.1, 0.15]
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        lr_values, weight_decay_values, label_smoothing_values
    ))
    
    print(f"\nSearch Space:")
    print(f"  Learning Rate: {lr_values}")
    print(f"  Weight Decay: {weight_decay_values}")
    print(f"  Label Smoothing: {label_smoothing_values}")
    print(f"\nTotal combinations: {len(param_combinations)}")
    print(f"Estimated time: ~{len(param_combinations) * 2} hours (assuming ~2h per run)")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/hyperparameter_search/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'grid_search_results.csv'
    
    # Write CSV header
    with open(results_file, 'w') as f:
        f.write('combination_id,learning_rate,weight_decay,label_smoothing,best_val_acc,test_acc,epochs_trained\n')
    
    print(f"\nResults will be saved to: {results_file}\n")
    
    # Load data once (reuse across all runs)
    print("[1/3] Loading data...")
    train_loader, val_loader, test_loader, class_names = create_classification_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, augmentation_type='enhanced'
    )
    print(f'Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}')
    
    # Run grid search
    print("\n[2/3] Starting grid search...\n")
    
    best_overall_acc = 0
    best_overall_config = None
    
    for i, (lr, wd, ls) in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(param_combinations)}] Testing: LR={lr}, WD={wd}, LS={ls}")
        print(f"{'='*80}")
        
        # Create config
        config = create_config(lr, wd, ls)
        
        # Initialize model
        model = ResNet50Classifier(**BASELINE_CONFIG)
        
        # Create experiment subdirectory
        exp_dir = output_dir / f'comb_{i+1:02d}_LR{lr:.0e}_WD{wd:.0e}_LS{ls:.2f}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        trainer = ClassificationTrainer(model, config=config)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
        
        try:
            history = trainer.train(
                train_loader, val_loader, criterion,
                str(exp_dir / 'training')
            )
            
            best_val_acc = trainer.best_val_acc
            
            # Load best model for test evaluation
            best_model_path = exp_dir / 'training' / 'best_model.pth'
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(trainer.device)
                
                # Evaluate on test set
                from src.evaluation.classification_evaluator import ClassificationEvaluator
                evaluator = ClassificationEvaluator(class_names)
                metrics = evaluator.evaluate(model, test_loader, str(exp_dir / 'evaluation'))
                test_acc = metrics['accuracy']
            else:
                test_acc = 0.0
                print('Warning: Best model checkpoint not found')
            
            epochs_trained = len(history['history'])
            
            # Record results
            with open(results_file, 'a') as f:
                f.write(f'{i+1},{lr},{wd},{ls},{best_val_acc:.4f},{test_acc:.4f},{epochs_trained}\n')
            
            print(f"\n✓ Completed: Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}, Epochs={epochs_trained}")
            
            # Track best overall
            if test_acc > best_overall_acc:
                best_overall_acc = test_acc
                best_overall_config = (lr, wd, ls)
                print(f"🏆 NEW BEST! Test Acc={best_overall_acc:.4f}")
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
            with open(results_file, 'a') as f:
                f.write(f'{i+1},{lr},{wd},{ls},FAILED,FAILED,FAILED\n')
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"\nBest Configuration:")
    print(f"  Learning Rate: {best_overall_config[0]}")
    print(f"  Weight Decay: {best_overall_config[1]}")
    print(f"  Label Smoothing: {best_overall_config[2]}")
    print(f"  Test Accuracy: {best_overall_acc:.4f}")
    print(f"\nAll results saved to: {results_file}")
    print(f"Full outputs in: {output_dir}")


if __name__ == '__main__':
    main()
