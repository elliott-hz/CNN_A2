"""
Training Package

Provides training frameworks for detection and classification models.
"""

# Classification trainer
from .ResNet50_trainer import (
    ClassificationTrainer,
    TrainingConfig,
    TRAINING_CONFIG_BASELINE,
    TRAINING_CONFIG_FC_V1,
    TRAINING_CONFIG_REDUCED_V1,
    TRAINING_CONFIG_REDUCED_V2,
    TRAINING_CONFIG_DEEPER_V1,
    TRAINING_CONFIG_DEEPER_V2,
    TRAINING_CONFIG_DEEPER_V3
)

# Detection trainers
from .YOLOv8_trainer import YOLOv8Trainer
from .FasterRCNN_trainer import FasterRCNNTainer

__all__ = [
    'ClassificationTrainer',
    'TrainingConfig',
    'TRAINING_CONFIG_BASELINE',
    'TRAINING_CONFIG_FC_V1',
    'TRAINING_CONFIG_REDUCED_V1',
    'TRAINING_CONFIG_REDUCED_V2',
    'TRAINING_CONFIG_DEEPER_V1',
    'TRAINING_CONFIG_DEEPER_V2',
    'TRAINING_CONFIG_DEEPER_V3',
    'YOLOv8Trainer',
    'FasterRCNNTrainer'
]
