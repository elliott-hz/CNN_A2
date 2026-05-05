"""
Faster R-CNN Trainer

Handles Faster R-CNN training process with comprehensive logging.
Supports CSV metrics logging for detailed epoch-by-epoch tracking.
"""

import torch
import csv
from pathlib import Path
from typing import Dict, Any, List

# ==============================================================================
# Training Configurations for Experiments V1, V2, V3
# ==============================================================================

FASTERRCNN_V1_CONFIG = {
    # Baseline Configuration
    'learning_rate': 0.001,
    'batch_size': 2,        # T4 GPU memory constraint (Faster R-CNN is memory-intensive)
    'epochs': 2,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'patience': 10,         # Early stopping patience
}

FASTERRCNN_V2_CONFIG = {
    # Deeper Backbone Configuration (Added Conv Layers)
    'learning_rate': 0.0005, # Lower LR for deeper model stability
    'batch_size': 2,         # Same batch size (deeper model uses slightly more memory)
    'epochs': 60,            # More epochs for convergence
    'optimizer': 'adam',
    'weight_decay': 5e-4,    # Higher weight decay to prevent overfitting
    'patience': 15,          # Longer patience for deeper model
}

FASTERRCNN_V3_CONFIG = {
    # Shallower Backbone Configuration (Reduced Conv Layers)
    'learning_rate': 0.001,
    'batch_size': 2,         # Can potentially increase but keeping consistent
    'epochs': 40,            # Fewer epochs needed for simpler model
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
        
        # Setup CSV logging
        csv_path = output_path / 'training_history.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.model.to(device)
        
        best_loss = float('inf')
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
            batch_count = 0
            skipped_batches = 0
            
            print(f"\nEpoch {epoch+1}/{self.epochs} - Training...")
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                # **DEBUG: Print first batch info to understand structure**
                if epoch == 0 and batch_idx == 0:
                    print(f"\n🔍 DEBUG: First batch structure:")
                    print(f"   Type of images: {type(images)}")
                    print(f"   Number of images: {len(images)}")
                    print(f"   Image shapes: {[img.shape for img in images]}")
                    print(f"   Type of targets: {type(targets)}")
                    print(f"   Number of targets: {len(targets)}")
                    if len(targets) > 0:
                        print(f"   Type of first target: {type(targets[0])}")
                        print(f"   Keys in first target: {targets[0].keys()}")
                        print(f"   Boxes shape: {targets[0]['boxes'].shape}")
                        print(f"   Labels shape: {targets[0]['labels'].shape}")
                
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # **CRITICAL: Skip batches where all images have no valid bboxes**
                # This can happen after bbox filtering. Empty targets cause loss_dict issues.
                if all(len(t['boxes']) == 0 for t in targets):
                    skipped_batches += 1
                    continue
                
                loss_dict = model(images, targets)
                
                # **Robust Error Handling: Check loss_dict type and handle edge cases**
                if isinstance(loss_dict, dict):
                    # Normal case: dict of losses
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, (list, tuple)):
                    # Edge case: returned as list/tuple (handle gracefully by skipping)
                    print(f"\n⚠️ Warning: Batch {batch_idx} returned {type(loss_dict)} instead of dict")
                    print(f"   Content: {loss_dict}")
                    print(f"   Targets boxes: {[len(t['boxes']) for t in targets]}")
                    skipped_batches += 1
                    continue
                else:
                    raise TypeError(f"Model returned unexpected type {type(loss_dict)}. Expected dict.")
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1
            
            # Calculate average loss (only from non-skipped batches)
            avg_train_loss = epoch_loss / max(batch_count, 1)
            
            # Report skipped batches if any
            if skipped_batches > 0:
                print(f"  ⚠ Skipped {skipped_batches} training batches (all images had no valid objects)")
            
            # Validation epoch
            model.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # **CRITICAL: Skip batches where all images have no valid bboxes**
                    if all(len(t['boxes']) == 0 for t in targets):
                        continue
                    
                    loss_dict = model(images, targets)
                    
                    # **Handle both dict and list/tuple cases**
                    if isinstance(loss_dict, dict):
                        losses = sum(loss for loss in loss_dict.values())
                        val_loss += losses.item()
                        val_batch_count += 1
                    elif isinstance(loss_dict, (list, tuple)):
                        # Skip this batch if it returns unexpected format
                        print(f"  ⚠️ Warning: Validation batch {batch_idx} returned {type(loss_dict)}, skipping")
                        continue
                    else:
                        raise TypeError(f"Model returned unexpected type {type(loss_dict)} in validation.")
            
            avg_val_loss = val_loss / max(val_batch_count, 1)
            
            # Log to CSV
            csv_writer.writerow([epoch + 1, f'{avg_train_loss:.4f}', 
                               f'{avg_val_loss:.4f}', f'{self.learning_rate:.8f}'])
            csv_file.flush()
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}] '
                      f'Train Loss: {avg_train_loss:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f}')
            
            # Checkpoint and early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
                
                # Save best model
                torch.save(model.model.state_dict(), output_path / 'best_model.pth')
                print(f'  ✓ New best model saved (val_loss: {best_loss:.4f})')
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f'\nEarly stopping triggered at epoch {epoch+1}')
                    break
        
        csv_file.close()
        
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
