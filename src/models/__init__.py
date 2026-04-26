"""
Models Module
Contains model definitions for detection and classification
"""

from .detection_model import YOLOv8Detector
from .classification_model import ResNet50Classifier, AlexNetClassifier, GoogLeNetClassifier
from .torchvision_detection import FasterRCNNDetector, SSDDetector

__all__ = [
    'YOLOv8Detector', 
    'ResNet50Classifier', 
    'AlexNetClassifier', 
    'GoogLeNetClassifier',
    'FasterRCNNDetector',
    'SSDDetector'
]
