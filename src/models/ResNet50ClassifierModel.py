"""
ResNet50 Classifier Model

Provides ResNet50 model with configurable parameters for bird classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier with configurable architecture.
    
    Supports:
    - Baseline: Single FC layer (2048 -> num_classes)
    - Customized: Multi-layer FC with BatchNorm (2048 -> 512 -> 256 -> num_classes)
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5, 
                 pretrained: bool = True, additional_fc_layers: bool = False,
                 use_batch_norm: bool = True):
        """
        Initialize ResNet50 classifier.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            pretrained: Use ImageNet pretrained weights
            additional_fc_layers: Add extra FC layers (customized version)
            use_batch_norm: Use batch normalization in custom layers
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Build classifier head
        if additional_fc_layers:
            # Customized: Multi-layer classifier
            layers = [
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512)
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(512))
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256)
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(256))
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            ])
            self.classifier = nn.Sequential(*layers)
        else:
            # Baseline: Simple single-layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
    
    def forward(self, x):
        """Forward pass."""
        # Extract features from backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply classifier
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_layer2: bool = False):
        """
        Unfreeze backbone for fine-tuning.
        
        Args:
            unfreeze_layer2: Also unfreeze layer2 (extended fine-tuning)
        """
        # Always unfreeze layer3 and layer4
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Optionally unfreeze layer2
        if unfreeze_layer2:
            for param in self.backbone.layer2.parameters():
                param.requires_grad = True
        
        # Unfreeze bn1
        for param in self.backbone.bn1.parameters():
            param.requires_grad = True


# Model configurations
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True
}

CUSTOMIZED_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.7,
    'pretrained': True,
    'additional_fc_layers': True,
    'use_batch_norm': True
}
