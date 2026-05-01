"""
Classification Evaluator

Evaluates trained models and generates metrics, reports, and visualizations.
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
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationEvaluator:
    """
    Evaluates classification models on test data.
    
    Responsibilities:
    - Generate predictions
    - Calculate metrics (accuracy, precision, recall, F1)
    - Create confusion matrix
    - Save evaluation results
    - Visualize training curves
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
    
    def plot_training_curves(self, history: List[Dict], output_dir: str):
        """
        Plot training and validation curves.
        
        Args:
            history: List of dictionaries containing epoch metrics
            output_dir: Directory to save plots
        """
        if not history:
            print("⚠ No training history available for plotting")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        train_accs = [h['train_acc'] for h in history]
        val_accs = [h['val_acc'] for h in history]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy curves
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Training curves saved to: {plot_path}')
    
    def analyze_overfitting(self, history: List[Dict]) -> Dict:
        """
        Analyze training history for overfitting/underfitting patterns.
        
        Args:
            history: List of dictionaries containing epoch metrics
            
        Returns:
            Dictionary with analysis results
        """
        if not history or len(history) < 5:
            return {
                'pattern': 'insufficient_data',
                'description': 'Not enough epochs to analyze patterns',
                'recommendation': 'Train for more epochs'
            }
        
        # Get final epochs metrics
        recent_epochs = history[-5:]  # Last 5 epochs
        avg_train_loss = np.mean([h['train_loss'] for h in recent_epochs])
        avg_val_loss = np.mean([h['val_loss'] for h in recent_epochs])
        avg_train_acc = np.mean([h['train_acc'] for h in recent_epochs])
        avg_val_acc = np.mean([h['val_acc'] for h in recent_epochs])
        
        # Check for overfitting
        loss_gap = avg_val_loss - avg_train_loss
        acc_gap = avg_train_acc - avg_val_acc
        
        # Detect trend
        val_losses_trend = [h['val_loss'] for h in history[-10:]]  # Last 10 epochs
        is_val_loss_increasing = len(val_losses_trend) > 1 and val_losses_trend[-1] > val_losses_trend[0]
        
        analysis = {
            'avg_train_loss': float(avg_train_loss),
            'avg_val_loss': float(avg_val_loss),
            'avg_train_acc': float(avg_train_acc),
            'avg_val_acc': float(avg_val_acc),
            'loss_gap': float(loss_gap),
            'acc_gap': float(acc_gap)
        }
        
        # Determine pattern
        if acc_gap > 0.15 or (is_val_loss_increasing and loss_gap > 0.2):
            analysis['pattern'] = 'overfitting'
            analysis['description'] = (
                f'Model shows signs of overfitting. '
                f'Training accuracy ({avg_train_acc:.4f}) is significantly higher than '
                f'validation accuracy ({avg_val_acc:.4f}). '
                f'Validation loss may be increasing.'
            )
            analysis['recommendation'] = (
                'Consider: 1) Add dropout, 2) Increase weight decay, '
                '3) Add data augmentation, 4) Use early stopping, 5) Reduce model complexity'
            )
        elif avg_train_acc < 0.6 and avg_val_acc < 0.6:
            analysis['pattern'] = 'underfitting'
            analysis['description'] = (
                f'Model shows signs of underfitting. '
                f'Both training ({avg_train_acc:.4f}) and validation ({avg_val_acc:.4f}) '
                f'accuracies are low.'
            )
            analysis['recommendation'] = (
                'Consider: 1) Train for more epochs, 2) Reduce regularization, '
                '3) Increase model capacity, 4) Lower learning rate, 5) Check data quality'
            )
        else:
            analysis['pattern'] = 'good_fit'
            analysis['description'] = (
                f'Model appears to have a good fit. '
                f'Training accuracy ({avg_train_acc:.4f}) and validation accuracy ({avg_val_acc:.4f}) '
                f'are reasonably close.'
            )
            analysis['recommendation'] = 'Model performance is acceptable. Continue monitoring.'
        
        return analysis
    
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
        output_path = Path(output_dir)
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
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, output_path)
        
        print(f'\nResults saved to: {output_path}')
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_path: Path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = output_path / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Confusion matrix saved to: {cm_path}')
