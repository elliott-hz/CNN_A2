"""
ResNet50 Classifier Model

Provides ResNet50 model with configurable parameters for bird classification.

Customization Options:
- Baseline: Standard ResNet50 with single FC layer
- Customized FC: Multi-layer FC head with BatchNorm
- Customized CNN: Modified backbone structure (add/remove conv layers)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any
import copy


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier with configurable architecture.
    
    Supports:
    - Baseline: Standard ResNet50 with single FC layer
    - Customized FC: Multi-layer FC with BatchNorm
    - Customized CNN: Modified backbone with structural changes
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5, 
                 pretrained: bool = True, additional_fc_layers: bool = False,
                 use_batch_norm: bool = True, modify_backbone: bool = False,
                 remove_layer: str = None, add_conv_after_layer: str = None):
        """
        Initialize ResNet50 classifier.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            pretrained: Use ImageNet pretrained weights
            additional_fc_layers: Add extra FC layers (customized version)
            use_batch_norm: Use batch normalization in custom layers
            modify_backbone: Enable backbone structural modifications
            remove_layer: Remove a backbone layer ('layer3' or 'layer4')
            add_conv_after_layer: Add conv block after specified layer ('layer1', 'layer2', 'layer3')
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.resnet50(weights=weights)
        
        # Apply backbone modifications if requested
        if modify_backbone:
            self.backbone = self._modify_backbone(base_model, remove_layer, add_conv_after_layer)
        else:
            self.backbone = base_model
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace final FC with Identity (we'll use our own classifier)
        self.backbone.fc = nn.Identity()
        
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
    
    def _modify_backbone(self, base_model, remove_layer=None, add_conv_after_layer=None):
        """
        Modify backbone structure for true CNN customization.
        
        Args:
            base_model: Original ResNet50 model
            remove_layer: Layer to remove ('layer3' or 'layer4')
            add_conv_after_layer: Add conv block after this layer
            
        Returns:
            Modified backbone model
        """
        modified = copy.deepcopy(base_model)
        
        # Option 1: Remove a layer (reduces depth)
        if remove_layer == 'layer3':
            # Skip layer3 entirely
            modified.layer3 = nn.Identity()
            print("✓ Backbone modification: Removed layer3")
        elif remove_layer == 'layer4':
            # Skip layer4 entirely  
            modified.layer4 = nn.Identity()
            print("✓ Backbone modification: Removed layer4")
        
        # Option 2: Add convolutional block after a layer (increases depth)
        if add_conv_after_layer:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._get_layer_channels(modified, add_conv_after_layer),
                    out_channels=self._get_layer_channels(modified, add_conv_after_layer),
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(self._get_layer_channels(modified, add_conv_after_layer)),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self._get_layer_channels(modified, add_conv_after_layer),
                    out_channels=self._get_layer_channels(modified, add_conv_after_layer),
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(self._get_layer_channels(modified, add_conv_after_layer)),
                nn.ReLU(inplace=True)
            )
            
            # Insert the conv block
            if add_conv_after_layer == 'layer1':
                original_layer1 = modified.layer1
                modified.layer1 = nn.Sequential(original_layer1, conv_block)
                print("✓ Backbone modification: Added conv block after layer1")
            elif add_conv_after_layer == 'layer2':
                original_layer2 = modified.layer2
                modified.layer2 = nn.Sequential(original_layer2, conv_block)
                print("✓ Backbone modification: Added conv block after layer2")
            elif add_conv_after_layer == 'layer3':
                original_layer3 = modified.layer3
                modified.layer3 = nn.Sequential(original_layer3, conv_block)
                print("✓ Backbone modification: Added conv block after layer3")
        
        return modified
    
    def _get_layer_channels(self, model, layer_name):
        """Get output channels from a specific layer."""
        if layer_name == 'layer1':
            return model.layer1[-1].conv3.out_channels
        elif layer_name == 'layer2':
            return model.layer2[-1].conv3.out_channels
        elif layer_name == 'layer3':
            return model.layer3[-1].conv3.out_channels
        elif layer_name == 'layer4':
            return model.layer4[-1].conv3.out_channels
        return 512  # default

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

# Baseline: Standard ResNet50
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': False
}

# Customized v1: Enhanced FC head only (NOT sufficient per teacher's requirements)
CUSTOMIZED_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.7,
    'pretrained': True,
    'additional_fc_layers': True,
    'use_batch_norm': True,
    'modify_backbone': False
}

# Customized v2: TRUE CNN customization - Add conv blocks after layer2
CUSTOMIZED_V2_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.6,
    'pretrained': True,
    'additional_fc_layers': True,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': None,
    'add_conv_after_layer': 'layer2'  # Add extra conv block after layer2
}

# Customized v3: Alternative - Remove layer3 to reduce depth
CUSTOMIZED_V3_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': 'layer3',  # Remove layer3 entirely
    'add_conv_after_layer': None
}
