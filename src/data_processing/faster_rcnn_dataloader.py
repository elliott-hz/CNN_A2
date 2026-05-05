"""
Faster R-CNN Dataset and DataLoader

Provides dataset classes and data loading utilities for Faster R-CNN training.
Supports COCO, Pascal VOC, and YOLO format annotations.
Optimized for consistent data splits across experiments.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torchvision.transforms.functional as F


class FasterRCNNDataset(Dataset):
    """
    Dataset class for Faster R-CNN.
    
    Converts annotations to the format expected by torchvision's Faster R-CNN:
    - images: List of tensors (C, H, W)
    - targets: List of dicts with keys:
        - boxes: FloatTensor[N, 4] in [x1, y1, x2, y2] format
        - labels: Int64Tensor[N]
        - image_id: Int64Tensor[1]
        - area: Tensor[N]
        - iscrowd: UInt8Tensor[N]
    """
    
    def __init__(self, 
                 image_dir: str,
                 annotation_dir: str,
                 annotation_format: str = 'coco',
                 transforms=None,
                 class_names: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            annotation_format: Format of annotations ('coco', 'pascal', 'yolo')
            transforms: Optional transforms to apply
            class_names: List of class names (excluding background)
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.annotation_format = annotation_format
        self.transforms = transforms
        self.class_names = class_names or []
        
        # Load image-annotation pairs
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {image_dir}")
        if self.class_names:
            print(f"Classes: {self.class_names}")
    
    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """Load image and annotation file pairs."""
        samples = []
        
        if self.annotation_format == 'coco':
            samples = self._load_coco_samples()
        elif self.annotation_format == 'pascal':
            samples = self._load_pascal_samples()
        elif self.annotation_format == 'yolo':
            samples = self._load_yolo_samples()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
        
        return samples
    
    def _load_coco_samples(self) -> List[Tuple[Path, Path]]:
        """Load samples from COCO format annotations."""
        # Find annotation JSON file
        annot_files = list(self.annotation_dir.glob('*.json'))
        if not annot_files:
            raise FileNotFoundError(f"No JSON annotation files found in {self.annotation_dir}")
        
        annot_file = annot_files[0]
        
        with open(annot_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image_id
        img_id_to_annots = {}
        for annot in coco_data['annotations']:
            img_id = annot['image_id']
            if img_id not in img_id_to_annots:
                img_id_to_annots[img_id] = []
            img_id_to_annots[img_id].append(annot)
        
        # Create samples list
        samples = []
        for img_id, filename in img_id_to_filename.items():
            if img_id in img_id_to_annots:
                img_path = self.image_dir / filename
                if img_path.exists():
                    samples.append((img_path, annot_file))
        
        return samples
    
    def _load_pascal_samples(self) -> List[Tuple[Path, Path]]:
        """Load samples from Pascal VOC format annotations."""
        # Get all XML annotation files
        annot_files = list(self.annotation_dir.glob('*.xml'))
        
        samples = []
        for annot_file in annot_files:
            # Convert .xml to .jpg/.png
            img_filename = annot_file.stem + '.jpg'
            img_path = self.image_dir / img_filename
            
            if img_path.exists():
                samples.append((img_path, annot_file))
        
        return samples
    
    def _load_yolo_samples(self) -> List[Tuple[Path, Path]]:
        """Load samples from YOLO format annotations."""
        # Get all txt annotation files
        annot_files = list(self.annotation_dir.glob('*.txt'))
        
        samples = []
        for annot_file in annot_files:
            # Convert .txt to .jpg/.png
            img_filename = annot_file.stem + '.jpg'
            img_path = self.image_dir / img_filename
            
            if img_path.exists():
                samples.append((img_path, annot_file))
        
        return samples
    
    def _parse_coco_annotation(self, annot_file: Path, img_filename: str) -> Dict:
        """Parse COCO format annotation for a specific image."""
        with open(annot_file, 'r') as f:
            coco_data = json.load(f)
        
        # Find image info
        img_info = None
        for img in coco_data['images']:
            if img['file_name'] == img_filename:
                img_info = img
                break
        
        if img_info is None:
            raise ValueError(f"Image {img_filename} not found in COCO annotations")
        
        # Get annotations for this image
        img_annots = [a for a in coco_data['annotations'] if a['image_id'] == img_info['id']]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for annot in img_annots:
            # COCO format: [x, y, width, height]
            x, y, w, h = annot['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Map category_id to label index (1-indexed, 0 is background)
            category_id = annot['category_id']
            # Assuming category_id matches class index + 1
            labels.append(category_id)
            
            areas.append(annot['area'])
            iscrowd.append(annot.get('iscrowd', 0))
        
        return {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8)
        }
    
    def _parse_pascal_annotation(self, annot_file: Path) -> Dict:
        """Parse Pascal VOC format annotation."""
        tree = ET.parse(annot_file)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Get class name and convert to index
            class_name = obj.find('name').text
            if class_name in self.class_names:
                label = self.class_names.index(class_name) + 1  # +1 because 0 is background
            else:
                label = 0  # Unknown class
            labels.append(label)
        
        # Calculate areas
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        
        return {
            'boxes': boxes_tensor,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([0]),  # Placeholder
            'area': areas,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.uint8)
        }
    
    def _parse_yolo_annotation(self, annot_file: Path, img_size: Tuple[int, int]) -> Dict:
        """Parse YOLO format annotation."""
        height, width = img_size
        
        boxes = []
        labels = []
        
        with open(annot_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert YOLO format (center_x, center_y, w, h) normalized to (x1, y1, x2, y2) absolute
                x1 = (x_center - w / 2) * width
                y1 = (y_center - h / 2) * height
                x2 = (x_center + w / 2) * width
                y2 = (y_center + h / 2) * height
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)  # +1 because 0 is background
        
        if len(boxes) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([0]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.uint8)
            }
        
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        
        return {
            'boxes': boxes_tensor,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'area': areas,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.uint8)
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.
        
        Returns:
            image: Tensor (C, H, W)
            target: Dict with boxes, labels, etc.
        """
        img_path, annot_path = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Parse annotation
        if self.annotation_format == 'coco':
            img_filename = img_path.name
            target = self._parse_coco_annotation(annot_path, img_filename)
        elif self.annotation_format == 'pascal':
            target = self._parse_pascal_annotation(annot_path)
        elif self.annotation_format == 'yolo':
            target = self._parse_yolo_annotation(annot_path, (img_height, img_width))
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
        
        # Apply transforms if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            # Default transform: convert to tensor
            img = F.to_tensor(img)
        
        return img, target


def create_faster_rcnn_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
    annotation_format: str = 'coco',
    class_names: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for Faster R-CNN.
    
    Args:
        data_root: Root directory containing train/valid/test folders
        batch_size: Batch size for training
        num_workers: Number of worker processes
        annotation_format: Format of annotations ('coco', 'pascal', 'yolo')
        class_names: List of class names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)
    
    # Define paths based on annotation format
    if annotation_format == 'coco':
        splits = ['train', 'valid', 'test']
        image_dirs = [data_root / split for split in splits]
        annot_dirs = [data_root / split for split in splits]
    elif annotation_format == 'pascal':
        splits = ['train', 'valid', 'test']
        image_dirs = [data_root / split for split in splits]
        annot_dirs = [data_root / split for split in splits]
    elif annotation_format == 'yolo':
        splits = ['train', 'valid', 'test']
        image_dirs = [data_root / split / 'images' for split in splits]
        annot_dirs = [data_root / split / 'labels' for split in splits]
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    
    # Create datasets
    train_dataset = FasterRCNNDataset(
        image_dir=str(image_dirs[0]),
        annotation_dir=str(annot_dirs[0]),
        annotation_format=annotation_format,
        class_names=class_names
    )
    
    val_dataset = FasterRCNNDataset(
        image_dir=str(image_dirs[1]),
        annotation_dir=str(annot_dirs[1]),
        annotation_format=annotation_format,
        class_names=class_names
    )
    
    test_dataset = FasterRCNNDataset(
        image_dir=str(image_dirs[2]),
        annotation_dir=str(annot_dirs[2]),
        annotation_format=annotation_format,
        class_names=class_names
    )
    
    # Custom collate function for Faster R-CNN
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
