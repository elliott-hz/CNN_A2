"""
YOLOv8 Detection Model

Provides YOLOv8 model wrapper for object detection.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, Any


class YOLOv8Detector(nn.Module):
    """
    YOLOv8 detector with configurable backbone size.
    """
    
    def __init__(self, backbone: str = 'm', input_size: int = 640, 
                 confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.45,
                 pretrained: bool = True):
        """
        Initialize YOLOv8 detector.
        
        Args:
            backbone: Model size ('n', 's', 'm', 'l', 'x')
            input_size: Input image size
            confidence_threshold: Detection confidence threshold
            nms_iou_threshold: NMS IoU threshold
            pretrained: Use pretrained weights
        """
        super(YOLOv8Detector, self).__init__()
        
        self.backbone = backbone
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        
        # Load YOLOv8 model
        model_name = f'yolov8{backbone}.pt' if pretrained else f'yolov8{backbone}.yaml'
        self.model = YOLO(model_name)
        
        # Set thresholds
        self.model.model.conf = confidence_threshold
        self.model.model.iou = nms_iou_threshold
    
    def forward(self, x, **kwargs):
        """Forward pass (for compatibility)."""
        return self.model(x, verbose=False, **kwargs)
    
    def train_model(self, data: str, epochs: int = 100, imgsz: int = None, **kwargs):
        """
        Train the model.
        
        Args:
            data: Dataset config path
            epochs: Number of epochs
            imgsz: Image size (uses self.input_size if None)
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        if imgsz is None:
            imgsz = self.input_size
        
        results = self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            **kwargs
        )
        return results
    
    def save(self, save_path: str):
        """Save model."""
        self.model.save(save_path)


# Model configurations
YOLOV8_BASELINE_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True
}
