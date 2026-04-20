"""
Detection Dataset Preprocessor
Converts raw detection dataset to unified format (X_train, X_valid, X_test, y_*)
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
import yaml
from tqdm import tqdm
import cv2
from typing import Tuple, List, Dict


class DetectionPreprocessor:
    """
    Preprocesses dog face detection dataset.
    Converts various annotation formats to unified numpy arrays.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['paths']['raw_data']) / "detection_dataset"
        self.processed_dir = Path(self.config['paths']['processed_data']) / "detection"
        self.image_size = self.config['datasets']['detection']['image_size']
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def is_processed(self) -> bool:
        """Check if data has already been preprocessed."""
        required_files = [
            self.processed_dir / "X_train.npy",
            self.processed_dir / "X_valid.npy",
            self.processed_dir / "X_test.npy",
            self.processed_dir / "y_train.npy",
            self.processed_dir / "y_valid.npy",
            self.processed_dir / "y_test.npy",
            self.processed_dir / "metadata.json"
        ]
        return all(f.exists() for f in required_files)
    
    def process(self):
        """Main preprocessing pipeline."""
        print("=" * 80)
        print("DETECTION DATASET PREPROCESSING")
        print("=" * 80)
        
        # Load and parse raw data
        print("\n[1/4] Loading raw dataset...")
        images, annotations = self._load_raw_data()
        print(f"  Loaded {len(images)} images with annotations")
        
        # Preprocess images and annotations
        print("\n[2/4] Preprocessing images and annotations...")
        X, y = self._preprocess_data(images, annotations)
        print(f"  Preprocessed shape: X={X.shape}, y={y.shape}")
        
        # Split dataset
        print("\n[3/4] Splitting dataset (70/20/10)...")
        splits = self._split_dataset(X, y)
        
        # Save processed data
        print("\n[4/4] Saving processed data...")
        self._save_splits(splits)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.processed_dir}")
        print(f"Total samples: {len(X)}")
        print(f"  Train: {splits['X_train'].shape[0]}")
        print(f"  Valid: {splits['X_valid'].shape[0]}")
        print(f"  Test: {splits['X_test'].shape[0]}")
    
    def _load_raw_data(self) -> Tuple[List, List]:
        """
        Load raw detection dataset.
        Handles different annotation formats (COCO, YOLO, CSV).
        
        Returns:
            Tuple of (image_paths, annotations)
        """
        images = []
        annotations = []
        
        # Try to detect annotation format
        # This is a simplified version - adapt based on actual dataset structure
        
        # Example: If COCO format
        coco_file = self.raw_data_dir / "annotations.json"
        if coco_file.exists():
            return self._load_coco_format(coco_file)
        
        # Example: If YOLO format
        labels_dir = self.raw_data_dir / "labels"
        if labels_dir.exists():
            return self._load_yolo_format(labels_dir)
        
        # Example: If CSV format
        csv_file = self.raw_data_dir / "annotations.csv"
        if csv_file.exists():
            return self._load_csv_format(csv_file)
        
        raise FileNotFoundError("Could not find annotations in supported formats (COCO/YOLO/CSV)")
    
    def _load_coco_format(self, coco_file: Path) -> Tuple[List, List]:
        """Load COCO format annotations."""
        import json
        
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        images = []
        annotations = []
        
        # Create image ID to path mapping
        img_id_to_path = {}
        for img_info in coco_data.get('images', []):
            img_id_to_path[img_info['id']] = self.raw_data_dir / img_info['file_name']
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann['bbox'])  # [x, y, w, h]
        
        # Build lists
        for img_id, img_path in img_id_to_path.items():
            if img_id in img_annotations:
                images.append(img_path)
                annotations.append(img_annotations[img_id])
        
        return images, annotations
    
    def _load_yolo_format(self, labels_dir: Path) -> Tuple[List, List]:
        """Load YOLO format annotations."""
        images = []
        annotations = []
        
        images_dir = self.raw_data_dir / "images"
        if not images_dir.exists():
            images_dir = self.raw_data_dir
        
        for label_file in labels_dir.glob("*.txt"):
            img_file = images_dir / label_file.stem
            if img_file.suffix == '':
                img_file = img_file.with_suffix('.jpg')
            
            if not img_file.exists():
                continue
            
            # Read YOLO annotations
            bboxes = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        _, x_c, y_c, w, h = map(float, parts[:5])
                        # Convert to absolute coordinates (will be resized later)
                        bboxes.append([x_c, y_c, w, h])
            
            if bboxes:
                images.append(img_file)
                annotations.append(bboxes)
        
        return images, annotations
    
    def _load_csv_format(self, csv_file: Path) -> Tuple[List, List]:
        """Load CSV format annotations."""
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        
        images = []
        annotations = {}
        
        for _, row in df.iterrows():
            img_path = self.raw_data_dir / row['filename']
            bbox = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
            
            if img_path not in annotations:
                images.append(img_path)
                annotations[img_path] = []
            
            annotations[img_path].append(bbox)
        
        return images, list(annotations.values())
    
    def _preprocess_data(self, images: List, annotations: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images and annotations.
        
        Args:
            images: List of image paths
            annotations: List of bounding box lists
            
        Returns:
            Tuple of (images_array, annotations_array)
        """
        X_list = []
        y_list = []
        
        for img_path, bboxes in tqdm(zip(images, annotations), total=len(images), desc="Processing"):
            try:
                # Load and resize image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize to target size
                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                
                # Normalize to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
                
                # Process annotations
                # Scale bounding boxes to match resized image
                original_h, original_w = img.shape[:2]
                scale_x = self.image_size / original_w
                scale_y = self.image_size / original_h
                
                scaled_bboxes = []
                for bbox in bboxes:
                    if len(bbox) == 4:
                        # Format: [x_min, y_min, x_max, y_max] or [x_c, y_c, w, h]
                        # Assuming normalized YOLO format
                        x_c, y_c, w, h = bbox
                        x_c_scaled = x_c * self.image_size
                        y_c_scaled = y_c * self.image_size
                        w_scaled = w * self.image_size
                        h_scaled = h * self.image_size
                        scaled_bboxes.append([x_c_scaled, y_c_scaled, w_scaled, h_scaled])
                
                X_list.append(img_rgb)
                y_list.append(scaled_bboxes if scaled_bboxes else [])
                
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(X_list)
        
        # For y, we'll save as object array since each image can have different number of boxes
        y = np.empty(len(y_list), dtype=object)
        for i, bboxes in enumerate(y_list):
            y[i] = np.array(bboxes) if bboxes else np.array([]).reshape(0, 4)
        
        return X, y
    
    def _split_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split dataset into train/valid/test sets.
        
        Args:
            X: Images array
            y: Annotations array
            
        Returns:
            Dictionary with split datasets
        """
        from sklearn.model_selection import train_test_split
        
        n_samples = len(X)
        train_ratio = self.config['datasets']['detection']['train_ratio']
        val_ratio = self.config['datasets']['detection']['val_ratio']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=1-train_ratio-val_ratio, random_state=42
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test
        }
    
    def _save_splits(self, splits: Dict[str, np.ndarray]):
        """Save split datasets to disk."""
        for key, data in splits.items():
            filepath = self.processed_dir / f"{key}.npy"
            np.save(filepath, data)
        
        # Save metadata
        metadata = {
            'total_samples': len(splits['X_train']) + len(splits['X_valid']) + len(splits['X_test']),
            'train_samples': len(splits['X_train']),
            'valid_samples': len(splits['X_valid']),
            'test_samples': len(splits['X_test']),
            'image_size': self.image_size,
            'split_ratios': {
                'train': self.config['datasets']['detection']['train_ratio'],
                'valid': self.config['datasets']['detection']['val_ratio'],
                'test': 1 - self.config['datasets']['detection']['train_ratio'] - self.config['datasets']['detection']['val_ratio']
            }
        }
        
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_split(self, split_name: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific data split.
        
        Args:
            split_name: 'train', 'valid', or 'test'
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = np.load(self.processed_dir / f"X_{split_name}.npy", allow_pickle=True)
        y = np.load(self.processed_dir / f"y_{split_name}.npy", allow_pickle=True)
        return X, y


if __name__ == "__main__":
    preprocessor = DetectionPreprocessor()
    if not preprocessor.is_processed():
        preprocessor.process()
    else:
        print("Data already preprocessed. Skipping.")
