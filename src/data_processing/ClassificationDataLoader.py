"""
Classification Data Loader

Provides unified data loading functionality for all classification experiments.
Ensures consistent dataset splits and configurable data augmentation strategies.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def create_classification_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    augmentation_type: str = 'none'
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Create train/val/test dataloaders with configurable augmentation.
    
    Args:
        data_root: Root directory containing train/valid/test folders
        batch_size: Batch size for dataloaders (optimized for T4 GPU)
        num_workers: Number of worker processes for data loading
        augmentation_type: Type of data augmentation to apply
            - 'none': No augmentation (default)
            - 'standard': Basic augmentation (for baseline experiments)
            - 'enhanced': Stronger augmentation (for customized experiments)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    
    Note:
        - Validation and test sets use the SAME transform (no augmentation)
        - This ensures fair comparison across experiments
        - Only training augmentation differs based on experiment type
    """
    
    # Define augmentation strategies
    if augmentation_type == 'none':
        # No augmentation - only basic preprocessing
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'standard':
        # Standard augmentation for baseline experiments
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'enhanced':
        # Enhanced augmentation for customized experiments
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}. Use 'none', 'standard', or 'enhanced'.")
    
    # Test/Validation transform (NO augmentation - consistent across all experiments)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(f'{data_root}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_root}/valid', transform=test_transform)
    test_dataset = datasets.ImageFolder(f'{data_root}/test', transform=test_transform)
    
    # Create dataloaders (T4 GPU optimized)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


# Convenience functions for specific augmentation types
def create_baseline_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 2):
    """Create dataloaders with standard augmentation (for baseline experiments)."""
    return create_classification_dataloaders(
        data_root, batch_size, num_workers, augmentation_type='standard'
    )


def create_enhanced_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 2):
    """Create dataloaders with enhanced augmentation (for customized experiments)."""
    return create_classification_dataloaders(
        data_root, batch_size, num_workers, augmentation_type='enhanced'
    )
