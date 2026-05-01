"""
Models Package

Provides detection and classification models.
"""

# Classification models
from .ResNet50ClassifierModel import (
    ResNet50Classifier, 
    BASELINE_CONFIG, 
    CUSTOMIZED_V1_CONFIG,
    CUSTOMIZED_V2_CONFIG,
    CUSTOMIZED_V3_CONFIG
)

# Detection models
from .YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_BASELINE_CONFIG
from .FasterRCNNDetectorModel import FasterRCNNDetector, FASTERRCNN_BASELINE_CONFIG

__all__ = [
    # Classification
    'ResNet50Classifier', 
    'BASELINE_CONFIG', 
    'CUSTOMIZED_V1_CONFIG',
    'CUSTOMIZED_V2_CONFIG',
    'CUSTOMIZED_V3_CONFIG',
    # Detection
    'YOLOv8Detector', 'YOLOV8_BASELINE_CONFIG',
    'FasterRCNNDetector', 'FASTERRCNN_BASELINE_CONFIG'
]
