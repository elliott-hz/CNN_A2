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
from typing import Dict, Any, Optional
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
                 remove_layer: Optional[str] = None, add_conv_after_layer: Optional[str] = None):
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
        
        # Validate parameters
        if remove_layer and remove_layer not in ['layer3', 'layer4']:
            raise ValueError(f"remove_layer must be 'layer3' or 'layer4', got '{remove_layer}'")
        
        if add_conv_after_layer and add_conv_after_layer not in ['layer1', 'layer2', 'layer3']:
            raise ValueError(f"add_conv_after_layer must be 'layer1', 'layer2', or 'layer3', got '{add_conv_after_layer}'")
        
        if remove_layer and add_conv_after_layer:
            raise ValueError("Cannot both remove and add layers simultaneously")
        
        # Load pretrained ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.resnet50(weights=weights)
        
        # Apply backbone modifications if requested
        if modify_backbone:
            self.backbone = self._modify_backbone(base_model, remove_layer, add_conv_after_layer)
        else:
            self.backbone = base_model
        
        # Get feature dimension AFTER modifications
        # Use dummy forward pass to determine actual output dimension
        num_features = self._get_feature_dimension()
        
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
    
    def _get_feature_dimension(self) -> int:
        """
        Determine the feature dimension after backbone modifications.
        
        Uses a dummy forward pass to get the actual output shape.
        
        Returns:
            Feature dimension (number of channels after global average pooling)
        """
        # Create a dummy input (batch_size=1 to save memory)
        dummy_input = torch.zeros(1, 3, 224, 224)
        
        # Temporarily set to eval mode to avoid BatchNorm issues with batch_size=1
        training_mode = self.backbone.training
        self.backbone.eval()
        
        with torch.no_grad():
            # Pass through backbone up to avgpool
            x = self.backbone.conv1(dummy_input)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            num_features = x.shape[1]
        
        # Restore training mode
        self.backbone.train(training_mode)
        
        return num_features
    
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
            # Skip layer3 entirely and adjust layer4 to accept layer2's output
            modified.layer3 = nn.Identity()
            
            # Modify layer4's first bottleneck to accept 512 channels instead of 1024
            # layer4[0].conv1 expects input from layer3 (normally 1024), but now gets 512 from layer2
            original_conv1 = modified.layer4[0].conv1
            modified.layer4[0].conv1 = nn.Conv2d(
                in_channels=512,  # Changed from 1024 to 512
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            # Also need to adjust the downsample if it exists
            if modified.layer4[0].downsample is not None:
                original_downsample_conv = modified.layer4[0].downsample[0]
                modified.layer4[0].downsample[0] = nn.Conv2d(
                    in_channels=512,  # Changed from 1024 to 512
                    out_channels=original_downsample_conv.out_channels,
                    kernel_size=original_downsample_conv.kernel_size,
                    stride=original_downsample_conv.stride,
                    padding=original_downsample_conv.padding,
                    bias=original_downsample_conv.bias is not None
                )
            
            print("✓ Backbone modification: Removed layer3 (adjusted layer4 to accept 512 channels)")
            
        elif remove_layer == 'layer4':
            # Skip layer4 entirely  
            modified.layer4 = nn.Identity()
            print("✓ Backbone modification: Removed layer4 (final features will be 1024 channels)")
        
        # Option 2: Add convolutional block after a layer (increases depth)
        if add_conv_after_layer:
            # Get channel count BEFORE wrapping (critical fix)
            num_channels = self._get_original_layer_channels(modified, add_conv_after_layer)
            
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(num_channels),
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
    
    def _get_original_layer_channels(self, model, layer_name):
        """
        Get output channels from a specific layer BEFORE any wrapping.
        
        Args:
            model: ResNet50 model (before modifications)
            layer_name: Layer name ('layer1', 'layer2', 'layer3', 'layer4')
            
        Returns:
            Number of output channels
        """
        # Access the last bottleneck block directly
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
        """
        Forward pass using backbone's built-in forward method.
        
        This approach is robust to backbone modifications (layer removal/addition).
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Use backbone's forward pass (handles all internal layers correctly)
        features = self.backbone(x)
        
        # Apply classifier head
        out = self.classifier(features)
        return out


# Model configurations

# Baseline: Standard ResNet50 with pretrained weights
BASELINE_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': False
}

# Customized v1: Enhanced FC head with stronger regularization
CUSTOMIZED_V1_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,                     # ↓ Reduced from 0.7 to 0.5 (less regularization)
    'pretrained': True,
    'additional_fc_layers': True,
    'use_batch_norm': True,
    'modify_backbone': False
}

# Customized v2: TRUE CNN - Added conv blocks after layer2 + enhanced FC
CUSTOMIZED_V2_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,                     # ↓ Reduced from 0.6 to 0.5 (less regularization)
    'pretrained': True,
    'additional_fc_layers': True,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': None,
    'add_conv_after_layer': 'layer2'
}

# Customized v3: TRUE CNN - Removed layer3 (reduced depth)
CUSTOMIZED_V3_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': 'layer3',
    'add_conv_after_layer': None
}

# Customized v4: TRUE CNN - Removed layer4 (alternative reduction)
CUSTOMIZED_V4_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': True,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': 'layer4',
    'add_conv_after_layer': None
}
