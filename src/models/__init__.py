"""
Models Package

Provides detection and classification models.
"""

# Classification models
from .ResNet50ClassifierModel import (
    ResNet50Classifier, 
    BASELINE_CONFIG, 
    CUSTOMIZED_FC_V1_CONFIG,
    CUSTOMIZED_REDUCED_V1_CONFIG,
    CUSTOMIZED_REDUCED_V2_CONFIG,
    CUSTOMIZED_DEEPER_V1_CONFIG,
    CUSTOMIZED_DEEPER_V2_CONFIG,
    CUSTOMIZED_DEEPER_V3_CONFIG
)

# Detection models
from .YOLOv8DetectorModel import YOLOv8Detector, YOLOV8_BASELINE_CONFIG
from .FasterRCNNDetectorModel import (
    FasterRCNNDetector, 
    FASTERRCNN_V1_CONFIG,
    FASTERRCNN_V2_CONFIG,
    FASTERRCNN_V3_CONFIG
)

__all__ = [
    # Classification
    'ResNet50Classifier', 
    'BASELINE_CONFIG', 
    'CUSTOMIZED_FC_V1_CONFIG',
    'CUSTOMIZED_REDUCED_V1_CONFIG',
    'CUSTOMIZED_REDUCED_V2_CONFIG',
    'CUSTOMIZED_DEEPER_V1_CONFIG',
    'CUSTOMIZED_DEEPER_V2_CONFIG',
    'CUSTOMIZED_DEEPER_V3_CONFIG',
    # Detection
    'YOLOv8Detector', 'YOLOV8_BASELINE_CONFIG',
    'FasterRCNNDetector', 
    'FASTERRCNN_V1_CONFIG',
    'FASTERRCNN_V2_CONFIG',
    'FASTERRCNN_V3_CONFIG'
]
