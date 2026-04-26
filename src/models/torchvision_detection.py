"""
Torchvision Detection Models
Faster R-CNN and SSD implementations using torchvision
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead
from typing import Dict, List, Tuple, Optional
import torchvision


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN detector with ResNet50+FPN backbone.
    
    Two-stage detection:
    1. Region Proposal Network (RPN) generates candidate boxes
    2. ROI heads classify and refine boxes
    
    Characteristics:
    - Higher accuracy than single-stage detectors
    - Slower inference speed
    - Better for small objects
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Whether to use pretrained COCO weights
        """
        super(FasterRCNNDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained Faster R-CNN
        if pretrained:
            self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Replace classifier for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Set model to training mode
        self.model.train()
    
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        Forward pass.
        
        Args:
            images: List of tensors, each representing an image
            targets: List of dicts with 'boxes' and 'labels' (only during training)
            
        Returns:
            During training: dict with losses
            During inference: list of dicts with detections
        """
        return self.model(images, targets)
    
    def predict(self, images: List[torch.Tensor], conf_threshold: float = 0.5):
        """
        Perform inference with confidence threshold filtering.
        
        Args:
            images: List of tensors
            conf_threshold: Minimum confidence score
            
        Returns:
            List of detection results
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] >= conf_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        self.model.train()
        return filtered_predictions
    
    def save(self, save_path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(load_path))


class SSDDetector(nn.Module):
    """
    SSD (Single Shot Detector) with VGG16 backbone.
    
    Single-stage detection:
    - Predicts bounding boxes and class probabilities in one pass
    - Uses multi-scale feature maps for different object sizes
    
    Characteristics:
    - Moderate inference speed
    - Good balance between speed and accuracy
    - Effective for small objects via multi-scale predictions
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize SSD detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Whether to use pretrained COCO weights
        """
        super(SSDDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained SSD
        if pretrained:
            self.model = ssd300_vgg16(weights='DEFAULT')
        else:
            self.model = ssd300_vgg16(weights=None)
        
        # Note: SSD model architecture is fixed, we can't easily change num_classes
        # For custom classes, you would need to retrain from scratch or fine-tune
        # The pretrained model has 91 classes (COCO), we'll fine-tune on our dataset
        
        # Set model to training mode
        self.model.train()
    
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        Forward pass.
        
        Args:
            images: List of tensors
            targets: List of dicts with 'boxes' and 'labels' (training only)
            
        Returns:
            During training: dict with losses
            During inference: list of dicts with detections
        """
        return self.model(images, targets)
    
    def predict(self, images: List[torch.Tensor], conf_threshold: float = 0.5):
        """
        Perform inference with confidence threshold filtering.
        
        Args:
            images: List of tensors
            conf_threshold: Minimum confidence score
            
        Returns:
            List of detection results
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] >= conf_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        self.model.train()
        return filtered_predictions
    
    def save(self, save_path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(load_path))


def create_faster_rcnn_model(num_classes: int = 2, pretrained: bool = True) -> FasterRCNNDetector:
    """Factory function to create Faster R-CNN model."""
    return FasterRCNNDetector(num_classes=num_classes, pretrained=pretrained)


def create_ssd_model(num_classes: int = 2, pretrained: bool = True) -> SSDDetector:
    """Factory function to create SSD model."""
    return SSDDetector(num_classes=num_classes, pretrained=pretrained)
