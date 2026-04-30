"""
YOLOv8 Trainer

Handles YOLOv8 training process using Ultralytics framework.
"""

from pathlib import Path
from typing import Dict, Any


class YOLOv8Trainer:
    """
    Simple trainer for YOLOv8 models.
    
    Responsibilities:
    - Configure and run YOLOv8 training
    - Save training results
    """
    
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 24,
                 epochs: int = 100, optimizer: str = 'adam', 
                 weight_decay: float = 1e-4, use_amp: bool = True):
        """
        Initialize YOLOv8 trainer.
        
        Args:
            learning_rate: Initial learning rate
            batch_size: Batch size
            epochs: Number of epochs
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            weight_decay: L2 regularization
            use_amp: Use mixed precision training
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.use_amp = use_amp
    
    def train(self, model, train_data: str, val_data: str, 
             output_dir: str, **kwargs) -> Dict:
        """
        Train YOLOv8 model.
        
        Args:
            model: YOLOv8Detector instance
            train_data: Training dataset config path
            val_data: Validation dataset config path
            output_dir: Directory to save outputs
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        print("=" * 80)
        print("YOLOv8 TRAINING")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare training arguments
        train_args = {
            'data': train_data,
            'epochs': self.epochs,
            'imgsz': model.input_size,
            'batch': self.batch_size,
            'lr0': self.learning_rate,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'amp': self.use_amp,
            'project': str(output_path),
            'name': 'train',
            'exist_ok': True,
        }
        
        # Add any additional kwargs
        train_args.update(kwargs)
        
        print(f"Training configuration:")
        print(f"  Dataset: {train_data}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Image size: {model.input_size}")
        print(f"  Output dir: {output_path}")
        
        # Run training
        print("\nStarting training...")
        results = model.train_model(**train_args)
        
        print(f"\nTraining completed!")
        print(f"Results saved to: {output_path}")
        
        return {'results': results, 'output_dir': output_path}
