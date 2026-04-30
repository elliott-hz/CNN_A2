"""
Classification Trainer

Handles the training loop: forward pass, backward pass, optimization, and validation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple


class ClassificationTrainer:
    """
    Simple trainer for classification models.
    
    Responsibilities:
    - Training loop with optimizer
    - Validation loop
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3, 
                 weight_decay: float = 1e-4, use_amp: bool = True):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_amp: Use mixed precision training
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = []
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             criterion: nn.Module, epochs: int, output_dir: str,
             scheduler=None, patience: int = 10) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            epochs: Number of epochs
            output_dir: Directory to save checkpoints
            scheduler: Learning rate scheduler (optional)
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        early_stop_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Record history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                early_stop_counter = 0
                
                checkpoint_path = output_path / 'best_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, checkpoint_path)
                print(f'  ✓ Best model saved (Val Acc: {val_acc:.4f})')
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
        
        return {'history': self.training_history, 'best_val_acc': self.best_val_acc}

if __name__ == "__main__":
    # Example usage
    from src.models.classification_model import ResNet50Classifier, BASELINE_CLASSIFICATION_CONFIG
    
    model_config = BASELINE_CLASSIFICATION_CONFIG
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'early_stopping_patience': 7,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'label_smoothing': 0.1,
        'class_weighting': True
    }
    
    trainer = ClassificationTrainer(model_config, training_config)
    model = ResNet50Classifier(model_config)
    
    # trainer.train(model, X_train, y_train, X_valid, y_valid, "outputs/test_run")
