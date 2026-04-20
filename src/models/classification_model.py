"""
Classification Model Definition
Single ResNet50Classifier class with configurable parameters for different variants
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier with configurable parameters.
    
    This class wraps the ResNet50 model and allows for different configurations
    through the config parameter, enabling different experiment variants
    (baseline, modified v1, modified v2) using the same class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ResNet50 classifier with configuration.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate before final layer (default 0.5)
                - pretrained: Whether to use pretrained ImageNet weights (default True)
                - freeze_backbone: Whether to freeze backbone layers initially (default True)
                - additional_fc_layers: Whether to add additional FC layers (default False)
                - use_batch_norm: Whether to use batch normalization in custom layers (default True)
        """
        super(ResNet50Classifier, self).__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', True)
        self.additional_fc_layers = config.get('additional_fc_layers', False)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # Load pretrained ResNet50
        if self.pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        if self.additional_fc_layers:
            # Add additional FC layers
            layers = []
            
            # First FC layer
            layers.append(nn.Linear(num_features, 512))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(512))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # Second FC layer
            layers.append(nn.Linear(512, 256))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(256))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # Final classification layer
            layers.append(nn.Linear(256, self.num_classes))
            
            self.classifier = nn.Sequential(*layers)
        else:
            # Simple single-layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, self.num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits
        """
        # Extract features using backbone (without final FC layer)
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
    
    def unfreeze_backbone(self, unfreeze_all: bool = False):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, only unfreeze later layers.
        """
        if unfreeze_all:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Only unfreeze later layers (layer3 and layer4)
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            
            # Also unfreeze bn1 if present
            for param in self.backbone.bn1.parameters():
                param.requires_grad = True
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, 
                     optimizer_type: str = 'adam'):
        """
        Create optimizer based on configuration.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
            
        Returns:
            PyTorch optimizer
        """
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        param_groups = [
            {'params': classifier_params, 'lr': lr},
        ]
        
        if backbone_params:
            # Use lower learning rate for backbone
            param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        
        # Create optimizer
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups, 
                weight_decay=weight_decay
            )
        else:  # Default to Adam
            optimizer = torch.optim.Adam(
                param_groups, 
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def save(self, save_path: str):
        """
        Save the model state dict and configuration.
        
        Args:
            save_path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'architecture': 'ResNet50Classifier'
        }
        torch.save(checkpoint, save_path)
    
    @classmethod
    def load(cls, load_path: str, map_location=None):
        """
        Load a saved model.
        
        Args:
            load_path: Path to saved model
            map_location: Device mapping for loading
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location=map_location)
        
        # Create new instance with saved config
        model = cls(checkpoint['config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def create_classification_model(config: Dict[str, Any]) -> ResNet50Classifier:
    """
    Factory function to create classification model instance.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        ResNet50Classifier instance
    """
    return ResNet50Classifier(config)


# Example configurations for different experiment variants
BASELINE_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.5,
    'pretrained': True,
    'freeze_backbone': True,
    'additional_fc_layers': False,
    'use_batch_norm': True
}

MODIFIED_V1_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.7,  # Higher dropout
    'pretrained': True,
    'freeze_backbone': True,
    'additional_fc_layers': True,  # Additional FC layers
    'use_batch_norm': True
}

MODIFIED_V2_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.3,  # Lower dropout
    'pretrained': True,
    'freeze_backbone': False,  # No freezing
    'additional_fc_layers': False,
    'use_batch_norm': True
}
