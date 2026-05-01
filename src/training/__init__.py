"""
Training Package

Provides training frameworks for detection and classification models.
"""

# Classification trainer
from .classification_trainer import (
    ClassificationTrainer,
    TrainingConfig,
    TRAINING_CONFIG_BASELINE,
    TRAINING_CONFIG_V1,
    TRAINING_CONFIG_V2,
    TRAINING_CONFIG_V3,
    TRAINING_CONFIG_V4
)

# Detection trainers
from .YOLOv8_trainer import YOLOv8Trainer
from .FasterRCNN_trainer import FasterRCNNTrainer

__all__ = [
    'ClassificationTrainer',
    'TrainingConfig',
    'TRAINING_CONFIG_BASELINE',
    'TRAINING_CONFIG_V1',
    'TRAINING_CONFIG_V2',
    'TRAINING_CONFIG_V3',
    'TRAINING_CONFIG_V4',
    'YOLOv8Trainer',
    'FasterRCNNTrainer'
]
