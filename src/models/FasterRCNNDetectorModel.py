"""
Faster R-CNN Detection Model

Provides Faster R-CNN model with ResNet50+FPN backbone for object detection.
Supports direct backbone customization (adding/removing convolutional layers).
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
from typing import Dict, Any, List, Optional


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN detector with ResNet50+FPN backbone and customization support.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 min_size: int = 640, max_size: int = 640,
                 customize_type: Optional[str] = None):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Use pretrained backbone
            min_size: Minimum image size
            max_size: Maximum image size
            customize_type: Type of customization ('deeper', 'shallower', or None for baseline)
        """
        super(FasterRCNNDetector, self).__init__()
        
        self.num_classes = num_classes
        self.customize_type = customize_type
        
        # Load pretrained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn_v2(
            pretrained=pretrained,
            weights_backbone="DEFAULT" if pretrained else None
        )
        
        # Apply customization before replacing classifier
        if customize_type == 'deeper':
            self._add_conv_layers()
        elif customize_type == 'shallower':
            self._reduce_conv_layers()
        
        # Replace classifier for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Set transform parameters
        self.model.transform.min_size = (min_size,)
        self.model.transform.max_size = max_size
    
    def _add_conv_layers(self):
        """
        Add extra convolutional layers to the ResNet50 backbone after layer2.
        Purpose: Deepen shallow-layer feature extraction for better fine-grained detection.
        
        Strategy: Wrap layer3 with a Sequential that first applies custom conv layers,
        then the original layer3. This preserves FPN's access to layer outputs.
        """
        try:
            # Access the ResNet50 backbone
            backbone = self.model.backbone.body
            
            device = next(self.model.parameters()).device
            
            # Get channel dimensions from layer2 output (should be 512)
            layer2_out_channels = backbone.layer2[-1].conv3.out_channels
            
            # Create new Conv-BN-ReLU blocks
            new_block = nn.Sequential(
                nn.Conv2d(layer2_out_channels, layer2_out_channels, kernel_size=3, 
                         stride=1, padding=1, bias=False),
                nn.BatchNorm2d(layer2_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer2_out_channels, layer2_out_channels, kernel_size=3,
                         stride=1, padding=1, bias=False),
                nn.BatchNorm2d(layer2_out_channels),
                nn.ReLU(inplace=True)
            ).to(device)
            
            # Store original layer3 and layer4
            original_layer3 = backbone.layer3
            original_layer4 = backbone.layer4
            
            # Replace layer3 with: custom_conv -> original_layer3
            # This way FPN can still access "layer3" output
            backbone.layer3 = nn.Sequential(
                new_block,
                original_layer3
            )
            
            print("✅ Added 2 convolutional layers to backbone after layer2 (deeper model)")
            print(f"   Layer structure preserved for FPN compatibility")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not add conv layers: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with standard model...")
    
    def _reduce_conv_layers(self):
        """
        Reduce convolutional layers in the ResNet50 backbone.
        Purpose: Create a lighter model with faster inference and reduced overfitting.
        Strategy: Remove some bottleneck blocks from layer3.
        """
        try:
            backbone = self.model.backbone.body
            
            # Standard ResNet50 layer3 has 6 bottleneck blocks
            # We'll reduce it to 3 blocks (remove 3 bottlenecks = 9 conv layers)
            original_num_blocks = len(backbone.layer3)
            new_num_blocks = max(1, original_num_blocks // 2)
            
            # Create new layer3 with fewer blocks
            # Get configuration from first block
            first_block = backbone.layer3[0]
            
            # Extract key parameters
            inplanes = first_block.conv1.in_channels
            planes = first_block.conv1.out_channels
            stride = first_block.downsample[0].stride if first_block.downsample else (1, 1)
            
            # Rebuild layer3 with reduced blocks
            new_layer3 = nn.Sequential()
            
            # First block might have downsampling
            for i in range(new_num_blocks):
                block_stride = stride if i == 0 else 1
                block = type(first_block)(
                    inplanes if i == 0 else planes * first_block.expansion,
                    planes,
                    stride=block_stride,
                    downsample=first_block.downsample if i == 0 else None
                )
                new_layer3.add_module(str(i), block)
            
            # Replace layer3
            backbone.layer3 = new_layer3
            
            print(f"✅ Reduced layer3 from {original_num_blocks} to {new_num_blocks} bottleneck blocks")
            print(f"✅ Removed {original_num_blocks - new_num_blocks} blocks ({(original_num_blocks - new_num_blocks) * 3} conv layers)")
            print("✅ Shallower backbone configured successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not reduce conv layers: {e}")
            print("Continuing with standard model...")
    
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
            'num_classes': self.num_classes,
            'customize_type': self.customize_type
        }, save_path)
    
    def load(self, load_path: str):
        """Load model weights."""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])


# Model configurations for 3 experiments
FASTERRCNN_V1_CONFIG = {
    'num_classes': 2,  # Will be updated dynamically
    'pretrained': True,
    'min_size': 640,
    'max_size': 640,
    'customize_type': None  # Baseline - no customization
}

FASTERRCNN_V2_CONFIG = {
    'num_classes': 2,  # Will be updated dynamically
    'pretrained': True,
    'min_size': 640,
    'max_size': 640,
    'customize_type': 'deeper'  # Adds conv layers to backbone
}

FASTERRCNN_V3_CONFIG = {
    'num_classes': 2,  # Will be updated dynamically
    'pretrained': True,
    'min_size': 640,
    'max_size': 640,
    'customize_type': 'shallower'  # Reduces conv layers in backbone
}
