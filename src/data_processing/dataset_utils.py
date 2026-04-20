"""
Dataset Utilities
Common utilities for loading and saving data
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def load_numpy_data(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed numpy data.
    
    Args:
        data_dir: Directory containing processed data
        split: Data split ('train', 'valid', 'test')
        
    Returns:
        Tuple of (X, y) arrays
    """
    data_path = Path(data_dir)
    X = np.load(data_path / f"X_{split}.npy", allow_pickle=True)
    y = np.load(data_path / f"y_{split}.npy", allow_pickle=True)
    return X, y


def save_numpy_data(data_dir: str, X: np.ndarray, y: np.ndarray, split: str = 'train'):
    """
    Save data as numpy arrays.
    
    Args:
        data_dir: Directory to save data
        X: Images array
        y: Labels/annotations array
        split: Data split name
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    np.save(data_path / f"X_{split}.npy", X)
    np.save(data_path / f"y_{split}.npy", y)


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4):
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Images array
        y: Labels array
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class NumpyDataset(Dataset):
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y
            self.transform = transform
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            image = self.X[idx]
            label = self.y[idx]
            
            if self.transform:
                # Apply transformation if needed
                pass
            
            return torch.FloatTensor(image), torch.tensor(label)
    
    dataset = NumpyDataset(X, y)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
