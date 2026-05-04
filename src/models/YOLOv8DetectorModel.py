"""
YOLOv8 Detection Model

Provides YOLOv8 model wrapper for object detection with support for
direct backbone customization (adding/removing convolutional layers).
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from typing import Dict, Any, Optional
import copy


class YOLOv8Detector(nn.Module):
    """
    YOLOv8 detector with configurable backbone size and customization support.
    """
    
    def __init__(self, backbone: str = 'm', input_size: int = 640, 
                 confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.45,
                 pretrained: bool = True, customize_type: Optional[str] = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            backbone: Model size ('n', 's', 'm', 'l', 'x')
            input_size: Input image size
            confidence_threshold: Detection confidence threshold
            nms_iou_threshold: NMS IoU threshold
            pretrained: Use pretrained weights
            customize_type: Type of customization ('deeper', 'shallower', or None for baseline)
        """
        super(YOLOv8Detector, self).__init__()
        
        self.backbone = backbone
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.customize_type = customize_type
        
        # Load YOLOv8 model (standard model first)
        model_name = f'yolov8{backbone}.pt' if pretrained else f'yolov8{backbone}.yaml'
        self.model = YOLO(model_name)
        
        # Apply customization after loading
        if customize_type == 'deeper':
            self._add_conv_layers()
        elif customize_type == 'shallower':
            self._reduce_conv_layers()
        
        # Set thresholds
        self.model.model.conf = confidence_threshold
        self.model.model.iou = nms_iou_threshold
    
    def _add_conv_layers(self):
        """
        Add extra convolutional layers to the backbone after layer2.
        Purpose: Deepen shallow-layer feature extraction for better fine-grained detection.
        """
        try:
            # Access the backbone model
            backbone = self.model.model.model[:10]  # YOLOv8m backbone indices 0-9
            
            # Identify the layer to insert after (index 2 in standard YOLOv8m)
            # Standard YOLOv8m backbone:
            # 0: Conv64 (P1/2)
            # 1: Conv128 (P2/4) 
            # 2: C2f[128, True] (depth=3)
            # 3: Conv256 (P3/8)
            # ...
            
            # Create new Conv layers to add (after backbone index 2)
            device = next(self.model.model.parameters()).device
            
            # Insert 2 Conv layers with 128 channels, 3x3 kernel, stride=1, padding=1
            # This maintains spatial resolution while deepening features
            new_conv_1 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True)
            ).to(device)
            
            new_conv_2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True)
            ).to(device)
            
            # Store original backbone layers
            original_layers = list(self.model.model.model.children())
            
            # Create new model with inserted layers
            # We need to rebuild the full backbone with the new layers
            new_layers = []
            
            # Copy layers 0-2 (keep as is)
            for i in range(3):
                new_layers.append(original_layers[i])
            
            # Add new conv layers
            new_layers.append(new_conv_1)
            new_layers.append(new_conv_2)
            
            # Copy remaining layers (adjust for the +2 indices)
            for i in range(3, len(original_layers)):
                new_layers.append(original_layers[i])
            
            # Replace the model's sequential with new layers
            self.model.model.model = nn.Sequential(*new_layers)
            
            print("✅ Added 2 convolutional layers to backbone (deeper model)")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not add conv layers: {e}")
            print("Continuing with standard model...")
    
    def _reduce_conv_layers(self):
        """
        Reduce convolutional layers in the backbone by modifying one C2f module.
        Purpose: Create a lighter model with faster inference and reduced overfitting.
        """
        try:
            # Access backbone sequential
            backbone = self.model.model.model
            
            # Identify the target C2f module to reduce. In current YOLOv8m,
            # backbone index 6 is the C2f module at P4/16 with 4 bottleneck repeats.
            target_idx = 6
            target_layer = backbone[target_idx]
            
            if not isinstance(target_layer, C2f):
                raise ValueError(f"Expected C2f at backbone index {target_idx}, found {type(target_layer)}")
            
            c1 = target_layer.cv1.conv.in_channels
            c2 = target_layer.cv2.conv.out_channels
            original_n = len(target_layer.m)
            
            # Compute expansion ratio e from the existing C2f config
            e = target_layer.cv1.conv.out_channels / (2 * c2)
            shortcut = getattr(target_layer.m[0], 'shortcut', False)
            g = target_layer.cv1.conv.groups
            
            # New reduced repeat count: half the original (4 -> 2)
            new_n = max(1, original_n // 2)
            reduced_layer = C2f(c1, c2, n=new_n, shortcut=shortcut, g=g, e=e)
            
            # Replace the layer in-place
            backbone[target_idx] = reduced_layer
            self.model.model.model = backbone
            
            print(f"✅ Replaced C2f at backbone index {target_idx}: original n={original_n}, new n={new_n}")
            print("✅ Shallower backbone configured successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not reduce conv layers: {e}")
            print("Continuing with standard model...")
    
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
    'pretrained': True,
    'customize_type': None  # Baseline - no customization
}

YOLOV8_V2_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'deeper'  # Adds conv layers to backbone
}

YOLOV8_V3_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'shallower'  # Reduces conv layers in backbone
}
