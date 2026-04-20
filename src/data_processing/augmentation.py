"""
Data Augmentation Utilities
Provides augmentation pipelines for detection and classification tasks
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_detection_augmentations(train: bool = True, image_size: int = 640):
    """
    Get augmentation pipeline for detection task.
    
    Args:
        train: Whether to use training augmentations
        image_size: Target image size
        
    Returns:
        Albumentations compose transform
    """
    if train:
        # Strong augmentations for training
        transform = A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    else:
        # Minimal augmentations for validation/test
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    return transform


def get_classification_augmentations(train: bool = True, image_size: int = 224):
    """
    Get augmentation pipeline for classification task.
    
    Args:
        train: Whether to use training augmentations
        image_size: Target image size
        
    Returns:
        Albumentations compose transform
    """
    if train:
        # Moderate augmentations for classification
        transform = A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        # Minimal augmentations for validation/test
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform


# Note: If albumentations is not available, here's a simple PyTorch-based alternative
def get_simple_classification_augmentations(train: bool = True):
    """
    Simple augmentation using torchvision (fallback if albumentations not installed).
    """
    from torchvision import transforms
    
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform
