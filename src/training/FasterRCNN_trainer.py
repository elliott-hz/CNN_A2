"""
Faster R-CNN Trainer

Handles Faster R-CNN training process using torchvision framework.
"""

import torch
from pathlib import Path
from typing import Dict, Any


class FasterRCNNTrainer:
    """
    Simple trainer for Faster R-CNN models.
    
    Responsibilities:
    - Training loop with optimizer
    - Model checkpointing
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
        Train Faster R-CNN model.
        
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
        print(f"  Device: {device}")
        
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
            
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stop_counter = 0
                
                checkpoint_path = output_path / 'best_model.pth'
                model.save(str(checkpoint_path))
                print(f'  ✓ Best model saved (Loss: {avg_loss:.4f})')
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
        
        return {'best_loss': best_loss, 'output_dir': output_path}
