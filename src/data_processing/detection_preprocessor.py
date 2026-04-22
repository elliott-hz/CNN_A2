"""
Detection Dataset Preprocessor with Albumentations Letterbox
Preprocesses detection dataset using letterbox resize to preserve aspect ratio.
Saves processed images as individual files (not numpy arrays) for memory efficiency.

Dataset Structure (YOLOv8 Standard Format):
data/processed/detection/
├── images/
│   ├── train/          # Training images (*.jpg)
│   ├── val/            # Validation images (*.jpg)
│   └── test/           # Test images (*.jpg)
├── labels/
│   ├── train/          # Training labels (*.txt, YOLO format)
│   ├── val/            # Validation labels (*.txt, YOLO format)
│   └── test/           # Test labels (*.txt, YOLO format)
├── dataset.yaml        # YOLOv8 configuration file
└── metadata.json       # Processing metadata

Input:
data/raw/detection_dataset/
├── train_img/          # Training images (*.jpg)
├── train_label/        # Training labels (*.txt, YOLO format)
├── val_img/           # Validation images (*.jpg)
└── val_label/         # Validation labels (*.txt, YOLO format)

Output:
- Processed images saved to: data/processed/detection/images/{train,val,test}/
- Annotations saved to: data/processed/detection/labels/{train,val,test}/
- Split metadata saved to: data/splitting/detection_split/
"""

import os
import shutil
from sklearn.model_selection import train_test_split
import yaml


class DetectionPreprocessor:
    """
    A preprocessor for detection datasets that organizes images and labels
    into train/val/test splits for YOLOv8 training.
    """
    
    def __init__(self, raw_data_path, processed_data_path="data/processed/detection", test_size=0.2, val_size=0.5):
        """
        Initializes the DetectionPreprocessor.
        
        Args:
            raw_data_path: Path to the raw dataset containing images and labels
            processed_data_path: Path to save the processed dataset
            test_size: Proportion of data to use for testing
            val_size: Proportion of remaining data to use for validation
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.test_size = test_size
        self.val_size = val_size
        
        # Create directory structure (YOLOv8 standard: images/split, labels/split)
        self.directories = {
            'train': {
                'images': os.path.join(self.processed_data_path, 'images', 'train'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'train')
            },
            'val': {
                'images': os.path.join(self.processed_data_path, 'images', 'val'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'val')
            },
            'test': {
                'images': os.path.join(self.processed_data_path, 'images', 'test'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'test')
            }
        }
        
        # Create all required directories
        for dirs in self.directories.values():
            os.makedirs(dirs['images'], exist_ok=True)
            os.makedirs(dirs['labels'], exist_ok=True)
    
    def _find_image_label_pairs(self):
        """
        Find all image-label pairs in the raw data directory.
        
        Returns:
            List of tuples (image_path, label_path) for valid pairs
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_dir = os.path.join(self.raw_data_path, 'train_img')  # Training images
        label_dir = os.path.join(self.raw_data_path, 'train_label')  # Training labels
        
        # Collect all training pairs
        pairs = self._collect_pairs(image_dir, label_dir, image_extensions)
        
        # Add validation data if it exists separately
        val_image_dir = os.path.join(self.raw_data_path, 'val_img')
        val_label_dir = os.path.join(self.raw_data_path, 'val_label')
        
        if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
            val_pairs = self._collect_pairs(val_image_dir, val_label_dir, image_extensions)
            pairs.extend(val_pairs)
        
        print(f"Found {len(pairs)} valid image-label pairs")
        return pairs
    
    def _collect_pairs(self, image_dir, label_dir, image_extensions):
        """
        Helper method to collect image-label pairs from specific directories.
        """
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            return []

        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        pairs = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Only add pairs where both image and label exist and label is not empty
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                pairs.append((img_path, label_path))
        
        return pairs
    
    def _split_data(self, pairs):
        """
        Split the data into train/val/test sets.
        
        Args:
            pairs: List of tuples (image_path, label_path)
            
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing the split pairs
        """
        # First, split into train+val and test
        train_val_pairs, test_pairs = train_test_split(
            pairs, 
            test_size=self.test_size, 
            random_state=42
        )
        
        # Then, split train+val into train and val
        train_pairs, val_pairs = train_test_split(
            train_val_pairs,
            test_size=self.val_size,
            random_state=42
        )
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def preprocess(self):
        """
        Performs the preprocessing by copying images and labels to the appropriate directories.
        """
        # Find all valid image-label pairs
        all_pairs = self._find_image_label_pairs()
        
        if len(all_pairs) == 0:
            raise ValueError("No valid image-label pairs found in the raw data directory.")
        
        # Split the data
        split_data = self._split_data(all_pairs)
        
        # Copy files to appropriate directories
        for split_name, pairs in split_data.items():
            print(f"Processing {split_name} data ({len(pairs)} pairs)...")
            
            for img_path, label_path in pairs:
                # Copy image
                img_filename = os.path.basename(img_path)
                new_img_path = os.path.join(self.directories[split_name]['images'], img_filename)
                shutil.copy2(img_path, new_img_path)
                
                # Copy label
                label_filename = os.path.basename(label_path)
                new_label_path = os.path.join(self.directories[split_name]['labels'], label_filename)
                shutil.copy2(label_path, new_label_path)
        
        # Generate YAML configuration file for YOLOv8
        self._generate_yaml_config()
        
        print(f"Preprocessing completed! Dataset saved to {self.processed_data_path}")
    
    def _generate_yaml_config(self):
        """
        Generates a YAML configuration file for YOLOv8 training.
        """
        yaml_config = {
            'path': self.processed_data_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (dog)
            'names': ['dog']  # Class names
        }
        
        yaml_path = os.path.join(self.processed_data_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"YAML configuration saved to {yaml_path}")


# Example usage:
if __name__ == "__main__":
    # Define paths
    raw_data_path = "data/raw/detection_dataset/"  # Path to your raw dataset
    processed_data_path = "data/processed/detection"  # Path for processed dataset (matches exp script expectation)
    
    # Create preprocessor instance
    preprocessor = DetectionPreprocessor(raw_data_path, processed_data_path)
    
    # Run preprocessing
    preprocessor.preprocess()
