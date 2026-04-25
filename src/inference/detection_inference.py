"""
Detection Inference
Standalone inference for detection model
"""

import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any
import cv2


class DetectionInference:
    """
    Inference pipeline for dog face detection.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize detection inference with trained model.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
        """
        self.model = YOLO(model_path)
        print(f"Loaded detection model from: {model_path}")
    
    def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.45) -> List[Dict[str, Any]]:
        """
        Predict bounding boxes in an image.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: NMS IoU threshold
            
        Returns:
            List of detections with bbox and confidence
        """
        # Run inference
        results = self.model(image_path, conf=conf, iou=iou)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class': int(boxes.cls[i].cpu().numpy())
                    }
                    detections.append(detection)
        
        return detections
    
    def predict_from_array(self, image_array, conf: float = 0.5, iou: float = 0.45) -> List[Dict[str, Any]]:
        """
        Predict bounding boxes from numpy array (BGR format).
        
        Args:
            image_array: Numpy array in BGR format (from OpenCV)
            conf: Confidence threshold
            iou: NMS IoU threshold
            
        Returns:
            List of detections with bbox and confidence
        """
        # Run inference directly on numpy array
        results = self.model(image_array, conf=conf, iou=iou)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class': int(boxes.cls[i].cpu().numpy())
                    }
                    detections.append(detection)
        
        return detections
    
    def visualize(self, image_path: str, output_path: str = None, **kwargs):
        """
        Visualize detection results on image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            **kwargs: Additional arguments for prediction
            
        Returns:
            Annotated image
        """
        # Run prediction and plot
        results = self.model(image_path, **kwargs)
        
        # Plot results
        plotted = results[0].plot()
        
        if output_path:
            cv2.imwrite(output_path, plotted)
            print(f"Visualization saved to: {output_path}")
        
        return plotted
