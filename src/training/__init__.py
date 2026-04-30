"""
Training Package

Provides training frameworks for detection and classification models.
"""

# Classification trainer
from .classification_trainer import ClassificationTrainer

# Detection trainers
from .YOLOv8_trainer import YOLOv8Trainer
from .FasterRCNN_trainer import FasterRCNNTrainer

__all__ = [
    'ClassificationTrainer',
    'YOLOv8Trainer',
    'FasterRCNNTrainer'
]
