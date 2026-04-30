"""
Detection Evaluator

Evaluates detection models and generates metrics and reports.
"""

import json
from pathlib import Path
from typing import Dict


class DetectionEvaluator:
    """
    Evaluates detection models on test data.
    
    Responsibilities:
    - Run model evaluation
    - Calculate mAP metrics
    - Save evaluation results
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_yolov8(self, model, test_dataset: str, output_dir: str) -> Dict:
        """
        Evaluate YOLOv8 model on test set.
        
        Args:
            model: YOLOv8Detector instance
            test_dataset: Test dataset config path
            output_dir: Directory to save results
            
        Returns:
            Evaluation metrics dictionary
        """
        print("=" * 80)
        print("YOLOv8 MODEL EVALUATION")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run validation using Ultralytics
        print(f"Evaluating on dataset: {test_dataset}")
        results = model.model.val(data=test_dataset)
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        # Print results
        print(f'\nEvaluation Metrics:')
        print(f'  mAP@0.5: {metrics["mAP50"]:.4f}')
        print(f'  mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall: {metrics["recall"]:.4f}')
        
        # Save results
        metrics_path = output_path / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f'\nResults saved to: {output_path}')
        
        return metrics
    
    def evaluate_fasterrcnn(self, model, test_loader, output_dir: str) -> Dict:
        """
        Evaluate Faster R-CNN model on test set.
        
        Args:
            model: FasterRCNNDetector instance
            test_loader: Test data loader
            output_dir: Directory to save results
            
        Returns:
            Evaluation metrics dictionary
        """
        print("=" * 80)
        print("FASTER R-CNN MODEL EVALUATION")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement proper Faster R-CNN evaluation
        # For now, return placeholder metrics
        metrics = {
            'mAP50': 0.0,
            'mAP50_95': 0.0,
            'note': 'Faster R-CNN evaluation not yet implemented'
        }
        
        print('Note: Faster R-CNN evaluation needs implementation')
        
        # Save results
        metrics_path = output_path / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
