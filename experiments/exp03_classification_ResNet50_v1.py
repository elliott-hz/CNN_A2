"""
Experiment 03: ResNet50 Baseline for Bird Classification

Simple pipeline:
1. Load data
2. Initialize model
3. Train
4. Evaluate
5. Save results
"""

import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ResNet50ClassifierModel import ResNet50Classifier, BASELINE_CONFIG
from src.training.classification_trainer import ClassificationTrainer
from src.evaluation.classification_evaluator import ClassificationEvaluator


def create_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 2):
    """Create train/val/test dataloaders - optimized for T4 GPU."""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    """Run Experiment 03: Baseline ResNet50."""
    
    print("=" * 80)
    print("EXPERIMENT 03: ResNet50 Baseline")
    print("=" * 80)
    
    # Configuration - Optimized for T4 GPU (16GB, ~10GB usable)
    STUDENT_ID = "25509225"
    DATA_ROOT = f"data/{STUDENT_ID}/Image_Classification/split_dataset"
    BATCH_SIZE = 16  # Reduced from 32 for T4 GPU memory constraints
    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 35
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-4
    
    # Create output directory with experiment name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = 'exp03_baseline'
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
    
    # Step 2: Initialize model (Baseline config)
    print("\n[2/5] Initializing model...")
    model = ResNet50Classifier(**BASELINE_CONFIG)
    model.freeze_backbone()  # Phase 1: freeze backbone
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')
    
    # Step 3: Train - Phase 1 (Frozen backbone)
    print("\n[3/5] Training Phase 1 (Frozen backbone)...")
    trainer1 = ClassificationTrainer(model, learning_rate=LR_PHASE1, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history1 = trainer1.train(
        train_loader, val_loader, criterion, EPOCHS_PHASE1,
        str(output_dir / 'phase1'), patience=10
    )
    print(f'Phase 1 Best Val Acc: {trainer1.best_val_acc:.4f}')
    
    # Step 4: Train - Phase 2 (Fine-tuning)
    print("\n[4/5] Training Phase 2 (Fine-tuning)...")
    model.unfreeze_backbone(unfreeze_layer2=False)  # Unfreeze layer3+layer4
    
    trainer2 = ClassificationTrainer(model, learning_rate=LR_PHASE2, weight_decay=1e-4)
    history2 = trainer2.train(
        train_loader, val_loader, criterion, EPOCHS_PHASE2,
        str(output_dir / 'phase2'), patience=10
    )
    print(f'Phase 2 Best Val Acc: {trainer2.best_val_acc:.4f}')
    
    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.evaluate(model, test_loader, str(output_dir / 'evaluation'))
    
    # Save experiment summary
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment 03: ResNet50 Baseline\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Configuration\n\n')
        f.write(f'- Architecture: ResNet50 (Baseline)\n')
        f.write(f'- Dropout: {BASELINE_CONFIG["dropout_rate"]}\n')
        f.write(f'- Additional FC Layers: {BASELINE_CONFIG["additional_fc_layers"]}\n')
        f.write(f'- Phase 1: Frozen backbone, {EPOCHS_PHASE1} epochs, LR={LR_PHASE1}\n')
        f.write(f'- Phase 2: Fine-tune layer3+4, {EPOCHS_PHASE2} epochs, LR={LR_PHASE2}\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- Best Val Accuracy: {trainer2.best_val_acc:.4f}\n')
        f.write(f'- Test Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'- Test F1 (macro): {metrics["f1_macro"]:.4f}\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
