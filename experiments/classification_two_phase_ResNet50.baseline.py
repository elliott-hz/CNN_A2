"""
Example: Two-Phase Training with Layer Freezing

This example demonstrates how to use freeze_backbone() and unfreeze_backbone()
methods for a two-phase training approach.

NOTE: This is NOT used in the current experiments (which use NO freezing),
but shows how you COULD use these methods if desired.
"""

import sys
from pathlib import Path
import torch
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, BASELINE_CONFIG
from src.training.classification_trainer import ClassificationTrainer, TrainingConfig
from src.data_processing.ClassificationDataLoader import create_baseline_dataloaders
from src.evaluation.classification_evaluator import ClassificationEvaluator


def print_trainable_status(model, phase_name):
    """Print which layers are trainable."""
    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print('='*60)
    
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    total_trainable = backbone_trainable + classifier_trainable
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Backbone trainable params: {backbone_trainable:,} ({backbone_trainable/total_params*100:.1f}%)")
    print(f"Classifier trainable params: {classifier_trainable:,} ({classifier_trainable/total_params*100:.1f}%)")
    print(f"Total trainable: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.1f}%)")
    print('='*60)


def main():
    """Demonstrate two-phase training with layer freezing."""
    
    print("=" * 80)
    print("EXAMPLE: Two-Phase Training with Layer Freezing")
    print("=" * 80)
    
    # Configuration
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/example_two_phase_training/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}\n')
    
    # Load data
    print("[1/4] Loading data...")
    train_loader, val_loader, test_loader, class_names = create_baseline_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE
    )
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = ResNet50Classifier(**BASELINE_CONFIG)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # ========== PHASE 1: Frozen Backbone ==========
    print("\n" + "="*80)
    print("PHASE 1: Training with FROZEN backbone")
    print("="*80)
    
    # Freeze backbone
    model.freeze_backbone()
    print_trainable_status(model, "After freeze_backbone()")
    
    # Phase 1 config (higher LR, fewer epochs)
    config_phase1 = TrainingConfig(
        learning_rate=1e-3,           # Higher LR for classifier
        weight_decay=1e-4,
        optimizer_type='adamw',
        epochs=15,                    # Short phase
        use_scheduler=False,
        use_early_stopping=True,
        early_stopping_patience=5,
        label_smoothing=0.1,
        use_amp=True,
        description='Phase 1: Training classifier with frozen backbone'
    )
    
    trainer1 = ClassificationTrainer(model, config=config_phase1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\nStarting Phase 1 training...")
    history1 = trainer1.train(
        train_loader, val_loader, criterion,
        str(output_dir / 'phase1')
    )
    
    print(f'\nPhase 1 Best Val Acc: {trainer1.best_val_acc:.4f}')
    
    # ========== PHASE 2: Unfrozen Backbone ==========
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning with UNFROZEN backbone")
    print("="*80)
    
    # Unfreeze backbone (layer3 + layer4 + bn1)
    model.unfreeze_backbone(unfreeze_layer2=False)
    print_trainable_status(model, "After unfreeze_backbone()")
    
    # Phase 2 config (lower LR, more epochs)
    config_phase2 = TrainingConfig(
        learning_rate=1e-5,           # Much lower LR for fine-tuning
        weight_decay=1e-4,
        optimizer_type='adamw',
        epochs=50,                    # Longer phase
        use_scheduler=True,
        scheduler_type='reduce_on_plateau',
        scheduler_patience=5,
        scheduler_factor=0.5,
        use_early_stopping=True,
        early_stopping_patience=15,
        label_smoothing=0.1,
        use_amp=True,
        description='Phase 2: Fine-tuning entire network with lower LR'
    )
    
    # Note: Need to recreate optimizer since we changed which params are trainable
    trainer2 = ClassificationTrainer(model, config=config_phase2)
    
    print("\nStarting Phase 2 training...")
    history2 = trainer2.train(
        train_loader, val_loader, criterion,
        str(output_dir / 'phase2')
    )
    
    print(f'\nPhase 2 Best Val Acc: {trainer2.best_val_acc:.4f}')
    
    # ========== EVALUATION ==========
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.evaluate(model, test_loader, str(output_dir / 'evaluation'))
    
    print(f'\nFinal Test Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Results saved to: {output_dir}')
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Phase 1 trains only the classifier (fast, prevents overfitting)")
    print("2. Phase 2 fine-tunes the entire network (better performance)")
    print("3. Use lower learning rate in Phase 2 to avoid destroying pretrained features")
    print("4. Current experiments don't use this approach (per teacher's requirement)")


if __name__ == '__main__':
    main()
