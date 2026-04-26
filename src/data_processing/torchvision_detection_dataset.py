"""
Torchvision Detection Dataset
Dataset loader for Faster R-CNN and SSD models using COCO JSON or VOC XML formats
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T


class DetectionDataset(Dataset):
    """
    Dataset for torchvision detection models.
    
    Supports both COCO JSON and VOC XML annotation formats.
    Returns images and targets in the format expected by Faster R-CNN and SSD.
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        annotations_dir: Optional[str] = None,
        is_voc: bool = False,
        transforms: Optional = None,
        min_size: int = 800,
        max_size: int = 1333
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            annotations_file: Path to COCO JSON file (for COCO format)
            annotations_dir: Directory containing VOC XML files (for VOC format)
            is_voc: Whether using VOC format (default False for COCO)
            transforms: Transforms to apply to images
            min_size: Minimum image size for resizing
            max_size: Maximum image size for resizing
        """
        self.image_dir = Path(image_dir)
        self.is_voc = is_voc
        self.min_size = min_size
        self.max_size = max_size
        
        # Setup transforms
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor(),
            ])
        else:
            self.transforms = transforms
        
        # Load annotations
        if is_voc:
            if annotations_dir is None:
                raise ValueError("annotations_dir must be provided for VOC format")
            self.annotations_dir = Path(annotations_dir)
            self.image_files, self.annotations = self._load_voc_annotations()
        else:
            if annotations_file is None:
                raise ValueError("annotations_file must be provided for COCO format")
            self.annotations_file = Path(annotations_file)
            self.image_files, self.annotations = self._load_coco_annotations()
        
        print(f"Loaded {len(self.image_files)} images from {image_dir}")
    
    def _load_coco_annotations(self):
        """Load annotations from COCO JSON format."""
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image ID to file name mapping
        img_id_to_filename = {}
        for img_info in coco_data['images']:
            img_id_to_filename[img_info['id']] = img_info['file_name']
        
        # Group annotations by image
        img_id_to_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_annotations:
                img_id_to_annotations[img_id] = []
            img_id_to_annotations[img_id].append(ann)
        
        # Build dataset
        image_files = []
        annotations = []
        
        for img_id, filename in img_id_to_filename.items():
            image_path = self.image_dir / filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image_files.append(filename)
            
            # Convert annotations to target format
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            
            if img_id in img_id_to_annotations:
                for ann in img_id_to_annotations[img_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    boxes.append([
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3]
                    ])
                    labels.append(ann['category_id'] + 1)  # +1 because background is class 0
                    areas.append(ann['area'])
                    iscrowd.append(ann.get('iscrowd', 0))
            
            annotations.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([img_id]),
                'area': torch.as_tensor(areas, dtype=torch.float32),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
            })
        
        return image_files, annotations
    
    def _load_voc_annotations(self):
        """Load annotations from VOC XML format."""
        # Get all XML files
        xml_files = list(self.annotations_dir.glob('*.xml'))
        
        image_files = []
        annotations = []
        
        for xml_file in xml_files:
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get filename
            filename_elem = root.find('filename')
            if filename_elem is None:
                continue
            filename = filename_elem.text
            
            image_path = self.image_dir / filename
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image_files.append(filename)
            
            # Extract bounding boxes
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            
            for obj in root.findall('object'):
                # Get class name and convert to ID
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                
                # For single class dataset, label is always 1
                # In multi-class, you'd need a class name to ID mapping
                label = 1  # Assuming single class + background
                
                # Get bounding box coordinates
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                
                area = (xmax - xmin) * (ymax - ymin)
                areas.append(area)
                iscrowd.append(0)
            
            annotations.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([len(image_files) - 1]),
                'area': torch.as_tensor(areas, dtype=torch.float32),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
            })
        
        return image_files, annotations
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            image: Tensor of shape (C, H, W)
            target: Dict with keys 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        # Load image
        img_path = self.image_dir / self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get target
        target = self.annotations[idx].copy()
        
        # Apply transforms
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target


def create_detection_dataloaders(
    dataset_config_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    model_type: str = 'faster_rcnn'
):
    """
    Create training, validation, and test dataloaders.
    
    Args:
        dataset_config_path: Path to dataset YAML config file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        model_type: 'faster_rcnn' or 'ssd' (affects image sizing)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import yaml
    
    # Load dataset config
    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    
    # Set image sizes based on model type
    if model_type == 'faster_rcnn':
        min_size, max_size = 800, 1333
    elif model_type == 'ssd':
        min_size, max_size = 300, 300
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Determine annotation format
    # Check if COCO or VOC format exists
    coco_annotations_dir = base_path / "annotations"
    voc_annotations_base = base_path / "annotations"
    
    # For now, assume COCO format (instances_*.json)
    # You can extend this to auto-detect format
    use_voc = False
    
    # Create datasets
    train_dataset = DetectionDataset(
        image_dir=str(base_path / config['train']),
        annotations_file=str(coco_annotations_dir / "instances_train.json"),
        is_voc=use_voc,
        min_size=min_size,
        max_size=max_size
    )
    
    val_dataset = DetectionDataset(
        image_dir=str(base_path / config['val']),
        annotations_file=str(coco_annotations_dir / "instances_val.json"),
        is_voc=use_voc,
        min_size=min_size,
        max_size=max_size
    )
    
    test_dataset = DetectionDataset(
        image_dir=str(base_path / config['test']),
        annotations_file=str(coco_annotations_dir / "instances_test.json"),
        is_voc=use_voc,
        min_size=min_size,
        max_size=max_size
    )
    
    # Custom collate function for detection models
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
