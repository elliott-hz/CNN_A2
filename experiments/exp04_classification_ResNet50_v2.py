"""
Experiment 04: Customized ResNet50 for Bird Classification

Modifications from baseline:
- Additional FC layers (2048 -> 512 -> 256 -> 10) with BatchNorm
- Higher dropout (0.7)
- Extended fine-tuning (unfreeze layer2+3+4)
- Stronger data augmentation
- Higher weight decay (5e-3)
"""

import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, CUSTOMIZED_CONFIG
from src.training.classification_trainer import ClassificationTrainer
from src.evaluation.classification_evaluator import ClassificationEvaluator


def create_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 2):
    """Create train/val/test dataloaders with enhanced augmentation - optimized for T4 GPU."""
    
    # Enhanced training transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased from 15
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # New
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(f'{data_root}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_root}/valid', transform=test_transform)
    test_dataset = datasets.ImageFolder(f'{data_root}/test', transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def main():
    """Run Experiment 04: Customized ResNet50."""
    
    print("=" * 80)
    print("EXPERIMENT 04: Customized ResNet50")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16  # Reduced from 32 for T4 GPU memory constraints
    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 45  # More epochs for customized model
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-4
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'exp04_customized'
    output_dir = Path(f'outputs/{experiment_name}/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nGPU: {gpu_name} ({gpu_memory:.1f} GB)')
        print(f'Recommended batch size for ResNet50 on this GPU: ≤16')
    else:
        print('\nWarning: CUDA not available, using CPU')
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE
    )
    print(f'Classes: {class_names}')
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    # Step 2: Initialize model (Customized config)
    print("\n[2/5] Initializing customized model...")
    print('Modifications:')
    print('  - Additional FC layers with BatchNorm')
    print('  - Higher dropout (0.7)')
    print('  - Enhanced data augmentation')
    
    model = ResNet50Classifier(**CUSTOMIZED_CONFIG)
    model.freeze_backbone()  # Phase 1: freeze backbone
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')
    
    # Step 3: Train - Phase 1 (Frozen backbone)
    print("\n[3/5] Training Phase 1 (Frozen backbone)...")
    trainer1 = ClassificationTrainer(model, learning_rate=LR_PHASE1, weight_decay=5e-3)  # Higher weight decay
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)  # Higher label smoothing
    
    history1 = trainer1.train(
        train_loader, val_loader, criterion, EPOCHS_PHASE1,
        str(output_dir / 'phase1'), patience=12
    )
    print(f'Phase 1 Best Val Acc: {trainer1.best_val_acc:.4f}')
    
    # Step 4: Train - Phase 2 (Extended fine-tuning)
    print("\n[4/5] Training Phase 2 (Extended fine-tuning)...")
    model.unfreeze_backbone(unfreeze_layer2=True)  # Unfreeze layer2+3+4 (extended)
    
    trainer2 = ClassificationTrainer(model, learning_rate=LR_PHASE2, weight_decay=5e-3)
    history2 = trainer2.train(
        train_loader, val_loader, criterion, EPOCHS_PHASE2,
        str(output_dir / 'phase2'), patience=12
    )
    print(f'Phase 2 Best Val Acc: {trainer2.best_val_acc:.4f}')
    
    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.evaluate(model, test_loader, str(output_dir / 'evaluation'))
    
    # Save experiment summary
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment 04: Customized ResNet50\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Modifications from Baseline\n\n')
        f.write(f'1. Additional FC layers: 2048 → 512 → 256 → 10 (with BatchNorm)\n')
        f.write(f'2. Higher dropout: {CUSTOMIZED_CONFIG["dropout_rate"]}\n')
        f.write(f'3. Extended fine-tuning: unfreeze layer2+3+4\n')
        f.write(f'4. Enhanced data augmentation\n')
        f.write(f'5. Higher weight decay: 5e-3\n')
        f.write(f'6. Higher label smoothing: 0.15\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Architecture: ResNet50 (Customized)\n')
        f.write(f'- Phase 1: Frozen backbone, {EPOCHS_PHASE1} epochs, LR={LR_PHASE1}\n')
        f.write(f'- Phase 2: Fine-tune layer2+3+4, {EPOCHS_PHASE2} epochs, LR={LR_PHASE2}\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- Best Val Accuracy: {trainer2.best_val_acc:.4f}\n')
        f.write(f'- Test Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'- Test F1 (macro): {metrics["f1_macro"]:.4f}\n\n')
        f.write(f'## Hypothesis\n\n')
        f.write(f'Additional layers enable learning complex feature combinations.\n')
        f.write(f'Extended unfreezing adapts lower-level features to bird-specific patterns.\n')
        f.write(f'Stronger regularization prevents overfitting despite increased capacity.\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
