"""
Pipeline Inference
End-to-end stacked inference: detection + classification
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .detection_inference import DetectionInference
from .classification_inference import ClassificationInference


class PipelineInference:
    """
    End-to-end inference pipeline combining detection and classification.
    
    Input: Image with dogs
    Output: Bounding boxes + emotion labels for each detected dog
    """
    
    def __init__(self, detection_model_path: str, classification_model_path: str,
                 class_names: list = None):
        """
        Initialize pipeline with both models.
        
        Args:
            detection_model_path: Path to detection model (.pt)
            classification_model_path: Path to classification model (.pth)
            class_names: List of emotion class names
        """
        # Initialize both models
        self.detector = DetectionInference(detection_model_path)
        self.classifier = ClassificationInference(classification_model_path, class_names)
        
        print("Pipeline inference initialized successfully")
    
    def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run end-to-end inference on an image.
        
        Args:
            image_path: Path to input image
            conf: Detection confidence threshold
            iou: NMS IoU threshold
            
        Returns:
            List of results, one per detected dog
        """
        # Step 1: Detect dog faces
        detections = self.detector.predict(image_path, conf=conf, iou=iou)
        
        if not detections:
            print("No dogs detected")
            return []
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Step 2: Classify emotion for each detected dog
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            # Crop face region with padding
            x1, y1, x2, y2 = map(int, bbox)
            h, w = original_img_rgb.shape[:2]
            
            # Add padding (10% of box size)
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * 0.1)
            pad_y = int(box_h * 0.1)
            
            x1_padded = max(0, x1 - pad_x)
            y1_padded = max(0, y1 - pad_y)
            x2_padded = min(w, x2 + pad_x)
            y2_padded = min(h, y2 + pad_y)
            
            # Crop face
            face_img = original_img_rgb[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Step 3: Classify emotion
            emotion_result = self.classifier.predict(face_img)
            
            # Combine results
            result = {
                'dog_id': i,
                'bbox': bbox,
                'detection_confidence': detection['confidence'],
                'emotion': emotion_result['predicted_class'],
                'emotion_confidence': emotion_result['confidence'],
                'emotion_probabilities': emotion_result['probabilities']
            }
            
            results.append(result)
        
        return results
    
    def visualize(self, image_path: str, output_path: str = None, **kwargs) -> np.ndarray:
        """
        Visualize pipeline results on image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            **kwargs: Additional arguments for prediction
            
        Returns:
            Annotated image as numpy array
        """
        # Run prediction
        results = self.predict(image_path, **kwargs)
        
        # Load image
        img = cv2.imread(image_path)
        
        # Draw results
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            emotion = result['emotion']
            det_conf = result['detection_confidence']
            emo_conf = result['emotion_confidence']
            
            label = f"{emotion} ({emo_conf:.2f})"
            det_label = f"det: {det_conf:.2f}"
            
            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            cv2.putText(img, label, (x1, y1 - 25), font, font_scale, color, thickness)
            cv2.putText(img, det_label, (x1, y1 - 5), font, 0.4, color, 1)
        
        # Save or return
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Visualization saved to: {output_path}")
        
        return img


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pipeline_inference.py <detection_model.pt> <classification_model.pth> [image.jpg]")
        sys.exit(1)
    
    detection_model = sys.argv[1]
    classification_model = sys.argv[2]
    image_path = sys.argv[3] if len(sys.argv) > 3 else "test_image.jpg"
    
    # Initialize pipeline
    pipeline = PipelineInference(detection_model, classification_model)
    
    # Run inference
    results = pipeline.predict(image_path)
    
    # Print results
    print("\nDetection Results:")
    for result in results:
        print(f"  Dog {result['dog_id']}:")
        print(f"    BBox: {result['bbox']}")
        print(f"    Emotion: {result['emotion']} (confidence: {result['emotion_confidence']:.2f})")
    
    # Visualize
    pipeline.visualize(image_path, "output_visualization.jpg")
