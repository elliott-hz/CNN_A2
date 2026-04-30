"""
Classification Evaluator

Evaluates trained models and generates metrics and reports.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm


class ClassificationEvaluator:
    """
    Evaluates classification models on test data.
    
    Responsibilities:
    - Generate predictions
    - Calculate metrics (accuracy, precision, recall, F1)
    - Create confusion matrix
    - Save evaluation results
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for reporting
        """
        self.class_names = class_names
        self.metrics = {}
    
    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, 
                output_dir: str) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            output_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Get predictions
        y_true, y_pred = self._get_predictions(model, test_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Compile metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        # Print results
        print(f'\nOverall Metrics:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision (weighted): {precision_weighted:.4f}')
        print(f'  Recall (weighted): {recall_weighted:.4f}')
        print(f'  F1-Score (weighted): {f1_weighted:.4f}')
        print(f'  Precision (macro): {precision_macro:.4f}')
        print(f'  Recall (macro): {recall_macro:.4f}')
        print(f'  F1-Score (macro): {f1_macro:.4f}')
        
        print(f'\nPer-Class Metrics:')
        for i, class_name in enumerate(self.class_names):
            print(f'  {class_name}: P={per_class_precision[i]:.4f} R={per_class_recall[i]:.4f} F1={per_class_f1[i]:.4f}')
        
        print(f'\nConfusion Matrix:\n{cm}')
        
        # Save results
        self._save_results(y_true, y_pred, output_dir)
        
        return self.metrics
    
    def _get_predictions(self, model: torch.nn.Module, 
                        test_loader: DataLoader) -> tuple:
        """
        Get model predictions.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Tuple of (true_labels, predicted_labels)
        """
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Handle GoogLeNet auxiliary outputs or other tuple returns
                if isinstance(outputs, tuple):
                    main_out = outputs[0]
                else:
                    main_out = outputs
                
                _, predicted = main_out.max(1)
                
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        
        return y_true, y_pred
    
    def _save_results(self, y_true, y_pred, output_dir: str):
        """
        Save evaluation results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dir: Directory to save results
        """
        output_path = Path(output_dir) / "logs"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = output_path / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save classification report
        report_text = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        
        report_path = output_path / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f'\nResults saved to: {output_path}')
