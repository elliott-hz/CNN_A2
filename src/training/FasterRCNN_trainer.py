"""
Faster R-CNN Trainer

Handles Faster R-CNN training process with comprehensive logging.
Supports CSV metrics logging for detailed epoch-by-epoch tracking.
"""

import torch
import csv
from pathlib import Path
from typing import Dict, Any, List


class FasterRCNNTrainer:
    """
    Enhanced trainer for Faster R-CNN models with CSV logging.
    
    Responsibilities:
    - Training loop with optimizer
    - Validation loop for monitoring
    - Model checkpointing
    - CSV metrics logging (epoch-by-epoch)
    """
    
    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize Faster R-CNN trainer.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def train(self, model, train_loader, val_loader, epochs: int, 
             output_dir: str, patience: int = 10) -> Dict:
        """
        Train Faster R-CNN model with CSV logging.
        
        Args:
            model: FasterRCNNDetector instance
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            output_dir: Directory to save outputs
            patience: Early stopping patience
            
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
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Device: {device}")
        print(f"  Early stopping patience: {patience}")
        print(f"  CSV logging: {csv_path}")
        
        for epoch in range(epochs):
            # Training epoch
            model.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation epoch
            val_loss = self._validate(model, val_loader, device)
            
            # Log to CSV
            current_lr = optimizer.param_groups[0]['lr']
            csv_writer.writerow([epoch + 1, f'{avg_train_loss:.6f}', 
                               f'{val_loss:.6f}', f'{current_lr:.8f}'])
            csv_file.flush()
            
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                
                checkpoint_path = output_path / 'best_model.pth'
                model.save(str(checkpoint_path))
                print(f'  ✓ Best model saved (Val Loss: {val_loss:.4f})')
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
        
        csv_file.close()
        print(f'\n✓ Training history saved to: {csv_path}')
        
        return {'best_loss': best_loss, 'output_dir': output_path}
    
    def _validate(self, model, val_loader, device) -> float:
        """
        Run validation loop.
        
        Args:
            model: FasterRCNNDetector instance
            val_loader: Validation data loader
            device: Computation device
            
        Returns:
            Average validation loss
        """
        model.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss
