"""
Faster R-CNN Trainer

Handles Faster R-CNN training process with comprehensive logging.
Supports CSV metrics logging for detailed epoch-by-epoch tracking.
"""

import torch
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# ==============================================================================
# Training Configurations for Experiments V1, V2, V3
# ==============================================================================

FASTERRCNN_V1_CONFIG = {
    # Baseline Configuration
    'learning_rate': 0.001,
    'batch_size': 2,        # T4 GPU memory constraint (Faster R-CNN is memory-intensive)
    'epochs': 1,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'patience': 10,         # Early stopping patience
}

FASTERRCNN_V2_CONFIG = {
    # Deeper Backbone Configuration (Added Conv Layers)
    'learning_rate': 0.0005, # Lower LR for deeper model stability
    'batch_size': 2,         # Same batch size (deeper model uses slightly more memory)
    'epochs': 1,            # More epochs for convergence
    'optimizer': 'adam',
    'weight_decay': 5e-4,    # Higher weight decay to prevent overfitting
    'patience': 15,          # Longer patience for deeper model
}

FASTERRCNN_V3_CONFIG = {
    # Shallower Backbone Configuration (Reduced Conv Layers)
    'learning_rate': 0.001,
    'batch_size': 2,         # Can potentially increase but keeping consistent
    'epochs': 1,            # Fewer epochs needed for simpler model
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'patience': 10,          # Standard patience
}


class FasterRCNNTrainer:
    """
    Enhanced trainer for Faster R-CNN models with CSV logging.
    
    Responsibilities:
    - Training loop with optimizer
    - Validation loop for monitoring
    - Model checkpointing
    - CSV metrics logging (epoch-by-epoch)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Faster R-CNN trainer with complete training configuration.
        
        Args:
            config: Training configuration dictionary containing:
                - learning_rate: Initial learning rate
                - weight_decay: L2 regularization
                - epochs: Number of training epochs
                - patience: Early stopping patience
                - optimizer: Optimizer type (currently only 'adam' supported)
        """
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.optimizer_type = config.get('optimizer', 'adam')
    
    def train(self, model, train_loader, val_loader, output_dir: str) -> Dict:
        """
        Train Faster R-CNN model with CSV logging.
        
        Args:
            model: FasterRCNNDetector instance
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save outputs
            
        Returns:
            Training history dictionary
        """
        print("=" * 80)
        print("FASTER R-CNN TRAINING")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging with core metrics
        csv_path = output_path / 'training_history.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 
            'train_loss', 
            'train_box_loss', 'train_cls_loss',
            'precision', 'recall',
            'map50', 'map50_95',
            'learning_rate'
        ])
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        
        # Setup learning rate scheduler (Cosine Annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.model.to(device)
        
        best_loss = float('inf')
        best_map50 = 0.0
        early_stop_counter = 0
        
        print(f"Training configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Device: {device}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"  CSV logging: {csv_path}")
        
        for epoch in range(self.epochs):
            # Training epoch
            model.model.train()
            epoch_loss = 0.0
            epoch_box_loss = 0.0
            epoch_cls_loss = 0.0
            batch_count = 0
            skipped_batches = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", dynamic_ncols=True, leave=True)
            
            for batch_idx, (images, targets) in enumerate(train_pbar):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Skip batches where all images have no valid bboxes
                if all(len(t['boxes']) == 0 for t in targets):
                    skipped_batches += 1
                    continue
                
                # Forward pass - returns loss dict in training mode
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Track box and cls loss components
                epoch_box_loss += loss_dict.get('loss_box_reg', 0).item()
                epoch_cls_loss += loss_dict.get('loss_classifier', 0).item()
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1
                
                # Update progress bar with current metrics
                avg_loss = epoch_loss / batch_count
                avg_box = epoch_box_loss / batch_count
                avg_cls = epoch_cls_loss / batch_count
                train_pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'box': f'{avg_box:.3f}',
                    'cls': f'{avg_cls:.3f}'
                })
            
            train_pbar.close()
            
            # Calculate average losses
            avg_train_loss = epoch_loss / max(batch_count, 1)
            avg_box_loss = epoch_box_loss / max(batch_count, 1)
            avg_cls_loss = epoch_cls_loss / max(batch_count, 1)
            
            # Report skipped batches if any
            if skipped_batches > 0:
                print(f"  ⚠ Skipped {skipped_batches} training batches (all images had no valid objects)")
            
            # Fast mAP evaluation on validation set (< 5 seconds)
            map50, map50_95, precision, recall = self._fast_evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{self.epochs} [Eval]  P={precision:.3f}, R={recall:.3f}, mAP@0.5={map50:.3f}, mAP@0.5:0.95={map50_95:.3f}")
            
            # Log to CSV with core metrics
            current_lr = scheduler.get_last_lr()[0]
            csv_writer.writerow([
                epoch + 1, 
                f'{avg_train_loss:.4f}', 
                f'{avg_box_loss:.4f}',
                f'{avg_cls_loss:.4f}',
                f'{precision:.4f}',
                f'{recall:.4f}',
                f'{map50:.4f}',
                f'{map50_95:.4f}',
                f'{current_lr:.8f}'
            ])
            csv_file.flush()
            
            # Step the learning rate scheduler after each epoch
            scheduler.step()
            
            # Checkpoint and early stopping (based on mAP@0.5 for better model selection)
            if map50 > best_map50:
                best_map50 = map50
                # Use train loss as a secondary metric for logging if needed, or just track mAP
                early_stop_counter = 0
                
                # Save best model
                torch.save(model.model.state_dict(), output_path / 'best_model.pth')
                print(f'  New best model saved (mAP@0.5: {map50:.3f})')
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f'\nEarly stopping triggered at epoch {epoch+1}')
                    break
        
        csv_file.close()
        
        # Generate training curve plots
        self._plot_training_curves(csv_path, output_path)
        
        history = {
            'best_loss': best_loss,
            'total_epochs': epoch + 1,
            'early_stopped': early_stop_counter >= self.patience
        }
        
        print(f"\nTraining completed!")
        print(f"  Best validation loss: {best_loss:.4f}")
        print(f"  Total epochs: {history['total_epochs']}")
        print(f"  Early stopped: {history['early_stopped']}")
        
        return history
    
    def _fast_evaluate(self, model, val_loader, device):
        """
        Fast mAP evaluation (< 5 seconds) using simplified IoU matching.
        
        Returns:
            tuple: (map50, map50_95, precision, recall)
        """
        all_preds = []
        all_gts = []
        
        # Collect predictions and ground truths with progress bar
        eval_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Eval]", dynamic_ncols=True, leave=False)
        
        with torch.no_grad():
            for images, targets in eval_pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Skip empty batches
                if all(len(t['boxes']) == 0 for t in targets):
                    continue
                
                model.model.eval()
                predictions = model(images)
                model.model.train()
                
                all_preds.extend(predictions)
                all_gts.extend(targets)
                
                # Update progress bar
                eval_pbar.set_postfix({'batch': len(images)})
        
        eval_pbar.close()
        
        if len(all_preds) == 0 or len(all_gts) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Compute fast mAP
        map50, map50_95, precision, recall = self._compute_fast_map(all_preds, all_gts)
        
        return map50, map50_95, precision, recall
    
    def _compute_fast_map(self, predictions, ground_truths, iou_thresholds=[0.5, 0.75]):
        """
        Compute approximate mAP using simplified IoU matching.
        
        Strategy:
        1. For each image, compute IoU between predictions and GT
        2. Match predictions to GT at IoU=0.5 and IoU=0.75
        3. Calculate Precision, Recall, and approximate AP
        4. Average across all images
        
        This is faster than full COCO mAP but captures the trend accurately.
        """
        total_tp_50 = 0
        total_fp_50 = 0
        total_fn_50 = 0
        
        total_tp_75 = 0
        total_fp_75 = 0
        total_fn_75 = 0
        
        total_gt_count = 0
        total_pred_count = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes'].cpu().numpy()
            gt_boxes = gt['boxes'].cpu().numpy()
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                total_fn_50 += len(gt_boxes)
                total_fn_75 += len(gt_boxes)
                total_gt_count += len(gt_boxes)
                total_pred_count += len(pred_boxes)
                continue
            
            # Compute IoU matrix
            ious = self._compute_iou_matrix(pred_boxes, gt_boxes)
            
            # For IoU=0.5
            tp_50, fp_50, fn_50 = self._match_predictions(ious, threshold=0.5)
            total_tp_50 += tp_50
            total_fp_50 += fp_50
            total_fn_50 += fn_50
            
            # For IoU=0.75
            tp_75, fp_75, fn_75 = self._match_predictions(ious, threshold=0.75)
            total_tp_75 += tp_75
            total_fp_75 += fp_75
            total_fn_75 += fn_75
            
            total_gt_count += len(gt_boxes)
            total_pred_count += len(pred_boxes)
        
        # Calculate metrics
        precision_50 = total_tp_50 / max(total_tp_50 + total_fp_50, 1)
        recall_50 = total_tp_50 / max(total_tp_50 + total_fn_50, 1)
        
        precision_75 = total_tp_75 / max(total_tp_75 + total_fp_75, 1)
        recall_75 = total_tp_75 / max(total_tp_75 + total_fn_75, 1)
        
        # Approximate AP (using single-point estimate)
        ap_50 = precision_50 * recall_50
        ap_75 = precision_75 * recall_75
        
        # mAP@0.5 = AP at IoU=0.5
        map50 = ap_50
        
        # mAP@0.5:0.95 ≈ average of AP at 0.5 and 0.75
        map50_95 = (ap_50 + ap_75) / 2.0
        
        # Use IoU=0.5 metrics for P and R
        precision = precision_50
        recall = recall_50
        
        return float(map50), float(map50_95), float(precision), float(recall)
    
    def _match_predictions(self, ious, threshold=0.5):
        """
        Match predictions to ground truth boxes based on IoU threshold.
        
        Args:
            ious: IoU matrix [N_pred, N_gt]
            threshold: IoU threshold for considering a match
            
        Returns:
            tuple: (tp, fp, fn)
        """
        n_pred, n_gt = ious.shape
        
        if n_pred == 0 or n_gt == 0:
            return 0, n_pred, n_gt
        
        # For each GT, find best matching prediction
        matched_gt = np.zeros(n_gt, dtype=bool)
        matched_pred = np.zeros(n_pred, dtype=bool)
        
        # Sort matches by IoU (greedy matching)
        matches = []
        for i in range(n_pred):
            for j in range(n_gt):
                if ious[i, j] >= threshold:
                    matches.append((ious[i, j], i, j))
        
        matches.sort(reverse=True)  # Highest IoU first
        
        tp = 0
        for iou_val, pred_idx, gt_idx in matches:
            if not matched_pred[pred_idx] and not matched_gt[gt_idx]:
                matched_pred[pred_idx] = True
                matched_gt[gt_idx] = True
                tp += 1
        
        fp = n_pred - tp
        fn = n_gt - tp
        
        return tp, fp, fn
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        Compute pairwise IoU between two sets of boxes.
        
        Args:
            boxes1: Array [N, 4] in (x1, y1, x2, y2) format
            boxes2: Array [M, 4] in (x1, y1, x2, y2) format
            
        Returns:
            IoU matrix [N, M]
        """
        N = len(boxes1)
        M = len(boxes2)
        
        if N == 0 or M == 0:
            return np.zeros((N, M))
        
        # Calculate intersection
        inter_x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0][np.newaxis, :])
        inter_y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1][np.newaxis, :])
        inter_x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2][np.newaxis, :])
        inter_y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3][np.newaxis, :])
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def _plot_training_curves(self, csv_path: Path, output_path: Path):
        """
        Generate simplified training curve plots (only 2 core charts).
        
        Args:
            csv_path: Path to training_history.csv
            output_path: Directory to save plots
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                print("⚠ Warning: No training data to plot")
                return
            
            # Plot 1: Training Loss Curves
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.plot(df['epoch'], df['train_loss'], 'b-o', label='Total Train Loss', linewidth=2, markersize=4)
            if 'train_box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train_box_loss'], 'g-s', label='Box Loss', linewidth=2, markersize=4)
            if 'train_cls_loss' in df.columns:
                ax1.plot(df['epoch'], df['train_cls_loss'], 'm-^', label='Cls Loss', linewidth=2, markersize=4)
            
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training Loss Components', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'loss_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Loss curve saved to: {output_path / 'loss_curve.png'}")
            
            # Plot 2: mAP Curve (most important!)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['epoch'], df['map50'], 'g-o', label='mAP@0.5', linewidth=2, markersize=6)
            ax.plot(df['epoch'], df['map50_95'], 'b-s', label='mAP@0.5:0.95', linewidth=2, markersize=6)
            
            # Also show P and R
            if 'precision' in df.columns and df['precision'].max() > 0:
                ax.plot(df['epoch'], df['precision'], 'm-^', label='Precision', 
                       linewidth=2, markersize=4, alpha=0.7)
            if 'recall' in df.columns and df['recall'].max() > 0:
                ax.plot(df['epoch'], df['recall'], 'c-d', label='Recall', 
                       linewidth=2, markersize=4, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_title('Core Performance Metrics', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'map_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ mAP curve saved to: {output_path / 'map_curve.png'}")
        
        except Exception as e:
            print(f"⚠ Warning: Failed to generate training plots: {e}")
