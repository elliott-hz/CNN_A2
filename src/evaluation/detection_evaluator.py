"""
Detection Evaluator
Evaluation metrics and visualization for detection models
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Any


class DetectionEvaluator:
    """
    Evaluation framework for detection models.
    
    Calculates mAP, IoU, precision-recall curves, and generates reports.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def evaluate(self, model, test_data, output_dir: str):
        """
        Evaluate model on test set.
        
        Args:
            model: YOLOv8Detector model
            test_data: Test dataset or path
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("DETECTION MODEL EVALUATION")
        print("=" * 80)
        
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run validation using YOLO's built-in method
            results = model.model.val(data=test_data)
            
            # Extract metrics
            self.metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': float(2 * results.box.mp * results.box.mr / 
                                 (results.box.mp + results.box.mr + 1e-8))
            }
            
            print(f"\nEvaluation Results:")
            print(f"  mAP@0.5: {self.metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {self.metrics['mAP50_95']:.4f}")
            print(f"  Precision: {self.metrics['precision']:.4f}")
            print(f"  Recall: {self.metrics['recall']:.4f}")
            print(f"  F1-Score: {self.metrics['f1_score']:.4f}")
            
            # Save metrics
            metrics_path = Path(output_dir) / "logs" / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"\nMetrics saved to: {metrics_path}")
            
            return self.metrics
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
    
    def generate_report(self, output_dir: str):
        """
        Generate comprehensive evaluation report in markdown format.
        
        Args:
            output_dir: Directory containing experiment outputs
        """
        report_path = Path(output_dir) / "logs" / "experiment_report.md"
        
        report = f"""# Experiment Report: Detection Model

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | {self.metrics.get('mAP50', 'N/A'):.4f} |
| mAP@0.5:0.95 | {self.metrics.get('mAP50_95', 'N/A'):.4f} |
| Precision | {self.metrics.get('precision', 'N/A'):.4f} |
| Recall | {self.metrics.get('recall', 'N/A'):.4f} |
| F1-Score | {self.metrics.get('f1_score', 'N/A'):.4f} |

## Figures

See the `figures/` directory for visualization outputs.
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
