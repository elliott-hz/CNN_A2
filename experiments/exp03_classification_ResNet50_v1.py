"""
Experiment 03: ResNet50 Baseline for Bird Classification

Simple pipeline:
1. Load data
2. Initialize model (ALL layers trainable - NO freezing per teacher's requirement)
3. Train with consistent methodology
4. Evaluate
5. Save results with training curves and analysis
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
    EPOCHS = 50  # Single-phase training (NO freezing)
    LR = 1e-4  # Lower LR since all layers are trainable from start
    
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
    
    # Step 2: Initialize model (Baseline config - ALL layers trainable)
    print("\n[2/5] Initializing model...")
    print('Methodology: Standard ResNet50 with ALL layers trainable (NO freezing)')
    print('This follows teacher\'s requirement: "If you freeze it, zero."')
    
    model = ResNet50Classifier(**BASELINE_CONFIG)
    # NOTE: NO freeze_backbone() call - all layers remain trainable
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)')
    
    # Step 3: Train (Single-phase, all layers trainable)
    print("\n[3/5] Training (Single-phase, all layers trainable)...")
    trainer = ClassificationTrainer(model, learning_rate=LR, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = trainer.train(
        train_loader, val_loader, criterion, EPOCHS,
        str(output_dir / 'training'), patience=10
    )
    print(f'Best Val Acc: {trainer.best_val_acc:.4f}')
    
    # Step 4: Evaluate on test set
    print("\n[4/5] Evaluating on test set...")
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.evaluate(model, test_loader, str(output_dir / 'evaluation'))
    
    # Plot training curves
    print("\n[5/5] Generating training curves and analysis...")
    evaluator.plot_training_curves(history['history'], str(output_dir / 'visualization'))
    
    # Analyze overfitting/underfitting
    analysis = evaluator.analyze_overfitting(history['history'])
    
    # Save experiment summary
    summary_path = output_dir / 'experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f'# Experiment 03: ResNet50 Baseline\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## Methodology\n\n')
        f.write(f'- **Architecture:** Standard ResNet50 with single FC layer (2048 → 10)\n')
        f.write(f'- **Training Strategy:** Single-phase training with ALL layers trainable\n')
        f.write(f'- **NO Layer Freezing:** Following teacher\'s requirement for correct methodology\n')
        f.write(f'- **Dropout:** {BASELINE_CONFIG["dropout_rate"]}\n')
        f.write(f'- **Epochs:** {EPOCHS}\n')
        f.write(f'- **Learning Rate:** {LR}\n')
        f.write(f'- **Weight Decay:** 1e-4\n')
        f.write(f'- **Label Smoothing:** 0.1\n')
        f.write(f'- **Batch Size:** {BATCH_SIZE} (T4 GPU optimized)\n\n')
        f.write(f'## Results\n\n')
        f.write(f'- Best Val Accuracy: {trainer.best_val_acc:.4f}\n')
        f.write(f'- Test Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'- Test F1 (macro): {metrics["f1_macro"]:.4f}\n')
        f.write(f'- Test Precision (weighted): {metrics["precision_weighted"]:.4f}\n')
        f.write(f'- Test Recall (weighted): {metrics["recall_weighted"]:.4f}\n\n')
        f.write(f'## Overfitting/Underfitting Analysis\n\n')
        f.write(f'**Pattern Detected:** {analysis["pattern"]}\n\n')
        f.write(f'{analysis["description"]}\n\n')
        f.write(f'**Recommendation:** {analysis["recommendation"]}\n\n')
        f.write(f'## Training Curves\n\n')
        f.write(f'See `visualization/training_curves.png` for:\n')
        f.write(f'- Training vs Validation Loss\n')
        f.write(f'- Training vs Validation Accuracy\n\n')
        f.write(f'## Key Design Decisions\n\n')
        f.write(f'1. **No Layer Freezing:** All layers trainable from start to ensure correct methodology\n')
        f.write(f'2. **Lower Learning Rate:** Started with 1e-4 instead of 1e-3 since all layers are training\n')
        f.write(f'3. **Single-Phase Training:** Simplified training process while maintaining effectiveness\n')
        f.write(f'4. **Consistent Dataset Split:** Same split used across all classification experiments\n')
    
    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_path}')
    print(f'Training curves: {output_dir / "visualization" / "training_curves.png"}')


if __name__ == '__main__':
    main()
