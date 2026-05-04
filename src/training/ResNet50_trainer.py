"""
Classification Trainer

Handles the training loop: forward pass, backward pass, optimization, and validation.
All training hyperparameters are managed through TrainingConfig objects.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
import csv
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    All training hyperparameters are centralized here for clarity and reproducibility.
    """
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_type: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    
    # Training schedule
    epochs: int = 50
    use_scheduler: bool = False
    scheduler_type: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'step', 'cosine'
    scheduler_patience: int = 5  # For ReduceLROnPlateau
    scheduler_factor: float = 0.5  # LR reduction factor
    
    # Learning rate warmup
    use_warmup: bool = False
    warmup_epochs: int = 5  # Number of warmup epochs
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Loss function
    label_smoothing: float = 0.1
    use_class_weights: bool = False
    
    # Mixed precision
    use_amp: bool = True
    
    # Description
    description: str = 'Default training configuration'


# Training configurations for Run-5 (Optimized based on Run-3 vs Run-4 analysis)

TRAINING_CONFIG_BASELINE = TrainingConfig(
    learning_rate=5e-4,
    weight_decay=5e-4, 
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,
    use_amp=True,
    description='Baseline RUN-5: Lighter regularization (WD=5e-4) to recover from Run-4 degradation'
)

TRAINING_CONFIG_V1 = TrainingConfig(
    learning_rate=5e-4,
    weight_decay=5e-4,                     
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                   
    use_amp=True,
    description='V1 RUN-5: No changes (already optimal, perfect consistency across runs)'
)

TRAINING_CONFIG_V2 = TrainingConfig(
    learning_rate=5e-4,
    weight_decay=5e-4,                     
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,
    use_amp=True,
    description='V2 RUN-5: No changes (already optimal, perfect consistency across runs)'
)

TRAINING_CONFIG_V3 = TrainingConfig(
    learning_rate=5e-4,                     
    weight_decay=5e-4,                      
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                  
    use_amp=True,
    description='V3 RUN-5: Moderate regularization (WD=2e-3) to potentially reach 97.5-97.8%'
)

# Training configuration for V4 (remove layer4) - Keep maximum regularization (successful fix)
TRAINING_CONFIG_V4 = TrainingConfig(
    learning_rate=5e-4,                     
    weight_decay=5e-4,                     
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                    
    use_amp=True,
    description='V4 RUN-5: Keep maximum regularization (fix was successful, no changes needed)'
)

# Training configuration for V5 (add conv blocks after layer1) - Option A: Intermediate WD
TRAINING_CONFIG_V5 = TrainingConfig(
    learning_rate=5e-4,                      
    weight_decay=5e-4,                      
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                
    use_amp=True,
    description='V5 RUN-5: Option A - Intermediate regularization (LR=1e-3, WD=2e-3, LS=0.10) targeting 97.3-97.8%'
)

# Training configuration for V6 (add conv blocks after layer2) - Lighter regularization
TRAINING_CONFIG_V6 = TrainingConfig(
    learning_rate=5e-4,                     
    weight_decay=5e-4,                       
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                    
    use_amp=True,
    description='V6 RUN-5: Lighter regularization (WD=1e-4) to recover from over-regularization (-2.01% drop)'
)

# Training configuration for V7 (add conv blocks after layer3) - Lighter regularization
TRAINING_CONFIG_V7 = TrainingConfig(
    learning_rate=5e-4,                     
    weight_decay=5e-4,                       
    optimizer_type='adamw',
    epochs=200,
    use_warmup=True,
    warmup_epochs=10,
    use_scheduler=True,
    scheduler_type='reduce_on_plateau',
    scheduler_patience=7,
    scheduler_factor=0.5,
    use_early_stopping=True,
    early_stopping_patience=50,
    label_smoothing=0.1,                     
    use_amp=True,
    description='V7 RUN-5: Lighter regularization (WD=1e-4) to recover from over-regularization (-1.81% drop)'
)

class ClassificationTrainer:
    """
    Simple trainer for classification models.
    
    Responsibilities:
    - Training loop with optimizer
    - Validation loop
    - Model checkpointing
    - CSV logging of training history
    - Model structure visualization
    
    All hyperparameters are controlled by TrainingConfig.
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        Initialize trainer with complete configuration.
        
        Args:
            model: PyTorch model to train
            config: Complete training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer based on config
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        if config.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Setup scheduler if enabled
        self.scheduler = None
        if config.use_scheduler:
            if config.scheduler_type == 'reduce_on_plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=config.scheduler_factor,
                    patience=config.scheduler_patience
                    # verbose parameter removed (deprecated in PyTorch 2.x)
                )
            elif config.scheduler_type == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.5
                )
            elif config.scheduler_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.epochs
                )
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = []
    
    def print_model_summary(self):
        """
        Print detailed model architecture summary.
        
        Shows each layer's name, input/output shapes, and parameter count.
        """
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*80)
        
        # Create a dummy input with batch_size=2 to satisfy BatchNorm requirements
        batch_size = 2
        channels = 3
        height = 224
        width = 224
        dummy_input = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # Set model to eval mode to avoid BatchNorm issues with small batches
        original_training = self.model.training
        self.model.eval()
        
        # Collect layer information
        layer_info = []
        total_params = 0
        trainable_params = 0
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__.__name__)
                
                # Skip containers
                if 'Sequential' in class_name or 'ModuleList' in class_name:
                    return
                
                # Get module name (use _get_name if available, else use type)
                if hasattr(module, '_get_name'):
                    module_name = module._get_name()
                else:
                    module_name = class_name
                
                # Count parameters
                module_params = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Get input/output shapes
                if isinstance(input, tuple):
                    input_shape = list(input[0].size())
                else:
                    input_shape = list(input.size())
                
                if isinstance(output, tuple):
                    output_shape = list(output[0].size())
                else:
                    output_shape = list(output.size())
                
                layer_info.append({
                    'name': module_name,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'params': module_params,
                    'trainable': module_trainable
                })
            
            if not list(module.children()):
                module.register_forward_hook(hook)
        
        # Register hooks
        self.model.apply(register_hook)
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Restore original training mode
        if original_training:
            self.model.train()
        
        # Print header
        print(f"\n{'Layer Name':<30} {'Input Shape':<20} {'Output Shape':<20} {'Params':<12} {'Trainable':<12}")
        print("-"*80)
        
        # Print each layer
        for info in layer_info:
            input_str = f"[{', '.join(map(str, info['input_shape']))}]"
            output_str = f"[{', '.join(map(str, info['output_shape']))}]"
            params_str = f"{info['params']:,}"
            trainable_str = f"{info['trainable']:,}"
            
            print(f"{info['name']:<30} {input_str:<20} {output_str:<20} {params_str:<12} {trainable_str:<12}")
            
            total_params += info['params']
            trainable_params += info['trainable']
        
        # Print summary
        print("-"*80)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"Model Size (approx): {total_params * 4 / (1024**2):.2f} MB")
        print("="*80 + "\n")

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
            if self.config.use_amp:
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
                
                if self.config.use_amp:
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
             criterion: nn.Module, output_dir: str) -> Dict:
        """
        Full training loop using configuration from self.config.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            output_dir: Directory to save checkpoints and logs
            
        Returns:
            Training history dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file for training history
        csv_path = output_path / 'training_history.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        
        early_stop_counter = 0
        epochs = self.config.epochs
        
        print(f'\nTraining Configuration:')
        print(f'  - Epochs: {epochs}')
        print(f'  - Learning Rate: {self.config.learning_rate}')
        if self.config.use_warmup:
            print(f'  - Warmup: Enabled ({self.config.warmup_epochs} epochs, linear)')
        print(f'  - Weight Decay: {self.config.weight_decay}')
        print(f'  - Optimizer: {self.config.optimizer_type.upper()}')
        print(f'  - Label Smoothing: {self.config.label_smoothing}')
        print(f'  - Early Stopping: {"Enabled" if self.config.use_early_stopping else "Disabled"} (patience={self.config.early_stopping_patience})')
        print(f'  - Scheduler: {"Enabled" if self.config.use_scheduler else "Disabled"}')
        print(f'  - Mixed Precision: {"Enabled" if self.config.use_amp else "Disabled"}')
        print(f'  - Description: {self.config.description}\n')
        
        try:
            for epoch in range(epochs):
                # Apply learning rate warmup if enabled
                if self.config.use_warmup and epoch < self.config.warmup_epochs:
                    # Linear warmup from 0 to base learning rate
                    warmup_ratio = (epoch + 1) / self.config.warmup_epochs
                    current_lr = self.config.learning_rate * warmup_ratio
                    
                    # Update learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, criterion, epoch)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader, criterion)
                
                # Update scheduler
                if self.scheduler is not None:
                    if self.config.scheduler_type == 'reduce_on_plateau':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Record history
                history_entry = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': current_lr
                }
                self.training_history.append(history_entry)
                
                # Write to CSV
                csv_writer.writerow([
                    epoch + 1,
                    f'{train_loss:.6f}',
                    f'{train_acc:.6f}',
                    f'{val_loss:.6f}',
                    f'{val_acc:.6f}',
                    f'{current_lr:.6f}'
                ])
                csv_file.flush()  # Ensure data is written
                
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
                        'epoch': epoch,
                        'config': self.config
                    }, checkpoint_path)
                    print(f'  ✓ Best model saved (Val Acc: {val_acc:.4f})')
                else:
                    early_stop_counter += 1
                
                # Early stopping
                if self.config.use_early_stopping and early_stop_counter >= self.config.early_stopping_patience:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    break
        finally:
            csv_file.close()
        
        print(f'\n✓ Training history saved to: {csv_path}')
        
        return {'history': self.training_history, 'best_val_acc': self.best_val_acc}