"""
YOLOv8 Trainer

Handles YOLOv8 training process using Ultralytics framework.
"""

from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# Training Configurations for Experiments V1, V2, V3
# ==============================================================================

YOLOV8_V1_CONFIG = {
    # Baseline Configuration
    'learning_rate': 0.001,
    'batch_size': 16,       # T4 GPU safe batch size
    'epochs': 5,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,        # Mixed precision
    'patience': 15,         # Early stopping patience
    'cos_lr': False,        # No cosine LR schedule for baseline
    'close_mosaic': 0,      # Keep mosaic augmentation throughout
}

YOLOV8_V2_CONFIG = {
    # Deeper Backbone Configuration (Added Conv Layers)
    'learning_rate': 0.0005, # Lower LR for deeper model stability
    'batch_size': 12,        # Smaller batch due to larger model memory usage
    'epochs': 5,           # More epochs for convergence
    'optimizer': 'adam',
    'weight_decay': 5e-4,    # Higher weight decay to prevent overfitting
    'use_amp': True,
    'patience': 20,          # Longer patience
    'cos_lr': True,          # Use cosine LR schedule for better convergence
    'close_mosaic': 10,      # Close mosaic in last 10 epochs
}

YOLOV8_V3_CONFIG = {
    # Shallower Backbone Configuration (Reduced Conv Layers)
    'learning_rate': 0.001,
    'batch_size': 20,        # Larger batch possible due to smaller model
    'epochs': 5,            # Fewer epochs needed for simpler model
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,
    'patience': 12,          # Shorter patience
    'cos_lr': False,
    'close_mosaic': 0,
}


class YOLOv8Trainer:
    """
    Simple trainer for YOLOv8 models.
    
    Responsibilities:
    - Configure and run YOLOv8 training
    - Save training results
    """
    
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 24,
                 epochs: int = 100, optimizer: str = 'adam', 
                 weight_decay: float = 1e-4, use_amp: bool = True,
                 patience: int = 15, cos_lr: bool = False, close_mosaic: int = 0):
        """
        Initialize YOLOv8 trainer.
        
        Args:
            learning_rate: Initial learning rate
            batch_size: Batch size
            epochs: Number of epochs
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            weight_decay: L2 regularization
            use_amp: Use mixed precision training
            patience: Early stopping patience
            cos_lr: Use cosine learning rate scheduler
            close_mosaic: Stop mosaic augmentation N epochs before end
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.patience = patience
        self.cos_lr = cos_lr
        self.close_mosaic = close_mosaic
    
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
            'name': 'train',  # This creates output_dir/train/ subdirectory
            'exist_ok': True,
            'patience': self.patience,
            'cos_lr': self.cos_lr,
            'close_mosaic': self.close_mosaic,
            'verbose': True,  # Enable verbose output for debugging
        }
        
        # Add any additional kwargs (allows overriding if needed)
        train_args.update(kwargs)
        
        print(f"Training configuration:")
        print(f"  Dataset: {train_data}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Image size: {model.input_size}")
        print(f"  Patience: {self.patience}")
        print(f"  Cosine LR: {self.cos_lr}")
        print(f"  Close Mosaic: {self.close_mosaic}")
        print(f"  Project (output base): {output_path}")
        print(f"  Name (subdirectory): train")
        print(f"  Expected output: {output_path / 'train'}")
        
        # Run training
        print("\nStarting training...")
        results = model.train_model(**train_args)
        
        # Verify output location
        expected_train_dir = output_path / 'train'
        if expected_train_dir.exists():
            print(f"\n✓ Training output verified at: {expected_train_dir}")
            # List key files
            if (expected_train_dir / 'results.csv').exists():
                print(f'  ✓ results.csv found')
            else:
                print(f'  ⚠ Warning: results.csv not found in {expected_train_dir}')
        else:
            print(f"\n⚠ Warning: Expected training output not found at: {expected_train_dir}")
            print(f"  Checking for alternative locations...")
            # Search for recent train directories
            import glob
            possible_locations = glob.glob(str(output_path / '**/train'), recursive=True)
            possible_locations += glob.glob('runs/detect/train*')
            if possible_locations:
                print(f"  Found potential locations: {possible_locations[:5]}")
        
        print(f"\nTraining completed!")
        print(f"Results should be at: {output_path}")
        
        return {'results': results, 'output_dir': output_path}
