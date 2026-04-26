"""
Training Module
Contains training frameworks for detection and classification models
"""

from .detection_trainer import DetectionTrainer
from .classification_trainer import ClassificationTrainer
from .torchvision_detection_trainer import TorchvisionDetectionTrainer

__all__ = ['DetectionTrainer', 'ClassificationTrainer', 'TorchvisionDetectionTrainer']