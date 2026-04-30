"""
Faster R-CNN Detection Model

Provides Faster R-CNN model with ResNet50+FPN backbone for object detection.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, Any, List


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN detector with ResNet50+FPN backbone.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 min_size: int = 640, max_size: int = 640):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Use pretrained backbone
            min_size: Minimum image size
            max_size: Maximum image size
        """
        super(FasterRCNNDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn_v2(
            pretrained=pretrained,
            weights_backbone="DEFAULT" if pretrained else None
        )
        
        # Replace classifier for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Set transform parameters
        self.model.transform.min_size = (min_size,)
        self.model.transform.max_size = max_size
    
    def forward(self, images: List[torch.Tensor], 
                targets: List[Dict[str, torch.Tensor]] = None):
        """
        Forward pass.
        
        Args:
            images: List of input tensors
            targets: Target dictionaries (training only)
            
        Returns:
            Training: dict with losses
            Inference: list of detections
        """
        return self.model(images, targets)
    
    def save(self, save_path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, save_path)
    
    def load(self, load_path: str):
        """Load model weights."""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])


# Model configurations
FASTERRCNN_BASELINE_CONFIG = {
    'num_classes': 2,  # 1 class + background
    'pretrained': True,
    'min_size': 640,
    'max_size': 640
}
