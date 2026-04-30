"""
Image Classification Dataset Splitting Script

This script splits the Image Classification dataset into train/validation/test sets
using the student ID as random seed for reproducibility.

Usage:
    python src/data_processing/classification_split.py
    
The script will:
1. Read the original dataset from data/25509225/Image_Classification/dataset/
2. Split it into train (70%), valid (15%), test (15%)
3. Save the split dataset to data/25509225/Image_Classification/split_dataset/
4. Generate a splitting report in outputs/
"""

import os
import sys
import shutil
import random
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ClassificationDatasetSplitter:
    """
    Splits image classification dataset into train/validation/test sets.
    
    Compatible with ResNet50 and custom CNN architectures using torchvision's
    ImageFolder format.
    """
    
    def __init__(self, student_id="25509225", 
                 train_ratio=0.70, 
                 valid_ratio=0.15, 
                 test_ratio=0.15):
        """
        Initialize the dataset splitter.
        
        Args:
            student_id: Student ID to use as random seed
            train_ratio: Ratio of training data (default: 0.70)
            valid_ratio: Ratio of validation data (default: 0.15)
            test_ratio: Ratio of test data (default: 0.15)
        """
        self.student_id = student_id
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        
        # Paths
        self.base_path = project_root / "data" / student_id / "Image_Classification"
        self.original_dataset_path = self.base_path / "dataset"
        self.split_dataset_path = self.base_path / "split_dataset"
        self.outputs_path = project_root / "outputs"
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Statistics
        self.stats = {}
        
    def verify_ratios(self):
        """Verify that split ratios sum to 1.0."""
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
    def get_image_files(self, class_path):
        """
        Get all image files in a class directory.
        
        Args:
            class_path: Path to class directory
            
        Returns:
            List of image file paths
        """
        image_files = []
        for f in os.listdir(class_path):
            if Path(f).suffix.lower() in self.image_extensions:
                image_files.append(f)
        return sorted(image_files)
    
    def split_class_images(self, images):
        """
        Split images into train/valid/test sets.
        
        Args:
            images: List of image filenames
            
        Returns:
            Tuple of (train_images, valid_images, test_images)
        """
        # Shuffle with fixed seed for reproducibility
        random.seed(int(self.student_id))
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_valid = int(n_total * self.valid_ratio)
        
        train_images = shuffled[:n_train]
        valid_images = shuffled[n_train:n_train + n_valid]
        test_images = shuffled[n_train + n_valid:]
        
        return train_images, valid_images, test_images
    
    def create_split_directory_structure(self):
        """Create the output directory structure."""
        splits = ['train', 'valid', 'test']
        for split in splits:
            split_path = self.split_dataset_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {split_path}")
    
    def copy_images_to_split(self, source_dir, dest_dir, images):
        """
        Copy images from source to destination directory.
        
        Args:
            source_dir: Source directory path
            dest_dir: Destination directory path
            images: List of image filenames to copy
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for img_name in images:
            src_path = source_dir / img_name
            dst_path = dest_dir / img_name
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⚠ Warning: File not found: {src_path}")
    
    def perform_split(self):
        """
        Perform the dataset splitting operation.
        
        Returns:
            Dictionary containing splitting statistics
        """
        print("="*80)
        print("IMAGE CLASSIFICATION DATASET SPLITTING")
        print("="*80)
        print(f"\nStudent ID: {self.student_id}")
        print(f"Original Dataset: {self.original_dataset_path}")
        print(f"Output Directory: {self.split_dataset_path}")
        print(f"Split Ratios - Train: {self.train_ratio:.0%}, Valid: {self.valid_ratio:.0%}, Test: {self.test_ratio:.0%}")
        print("\n" + "-"*80)
        
        # Verify ratios
        self.verify_ratios()
        
        # Check if original dataset exists
        if not self.original_dataset_path.exists():
            raise FileNotFoundError(f"Original dataset not found at: {self.original_dataset_path}")
        
        # Get all classes
        classes = sorted([d.name for d in self.original_dataset_path.iterdir() if d.is_dir()])
        print(f"\nFound {len(classes)} classes:")
        for i, cls in enumerate(classes, 1):
            print(f"  {i}. {cls}")
        
        print("\n" + "-"*80)
        print("Processing each class...")
        print("-"*80 + "\n")
        
        # Create output directory structure
        self.create_split_directory_structure()
        
        # Process each class
        total_stats = {
            'classes': {},
            'totals': {'train': 0, 'valid': 0, 'test': 0, 'total': 0}
        }
        
        for cls in classes:
            cls_path = self.original_dataset_path / cls
            
            # Get all images for this class
            images = self.get_image_files(cls_path)
            
            if not images:
                print(f"⚠ Warning: No images found for class '{cls}', skipping...")
                continue
            
            # Split images
            train_imgs, valid_imgs, test_imgs = self.split_class_images(images)
            
            # Copy images to respective directories
            for split_name, split_imgs in [('train', train_imgs), 
                                           ('valid', valid_imgs), 
                                           ('test', test_imgs)]:
                dest_dir = self.split_dataset_path / split_name / cls
                self.copy_images_to_split(cls_path, dest_dir, split_imgs)
                
                # Update statistics
                total_stats['totals'][split_name] += len(split_imgs)
            
            total_stats['totals']['total'] += len(images)
            total_stats['classes'][cls] = {
                'total': len(images),
                'train': len(train_imgs),
                'valid': len(valid_imgs),
                'test': len(test_imgs)
            }
            
            print(f"✓ {cls:30s} | Total: {len(images):4d} | "
                  f"Train: {len(train_imgs):4d} | Valid: {len(valid_imgs):4d} | Test: {len(test_imgs):4d}")
        
        self.stats = total_stats
        
        print("\n" + "="*80)
        print("SPLITTING SUMMARY")
        print("="*80)
        print(f"\nTotal Images: {total_stats['totals']['total']}")
        print(f"  Training:   {total_stats['totals']['train']} ({total_stats['totals']['train']/total_stats['totals']['total']:.1%})")
        print(f"  Validation: {total_stats['totals']['valid']} ({total_stats['totals']['valid']/total_stats['totals']['total']:.1%})")
        print(f"  Testing:    {total_stats['totals']['test']} ({total_stats['totals']['test']/total_stats['totals']['total']:.1%})")
        print(f"\nOutput saved to: {self.split_dataset_path}")
        print("="*80)
        
        return total_stats
    
    def generate_split_report(self):
        """Generate a JSON report of the splitting operation."""
        if not self.stats:
            print("⚠ No statistics available. Run perform_split() first.")
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'student_id': self.student_id,
            'split_ratios': {
                'train': self.train_ratio,
                'valid': self.valid_ratio,
                'test': self.test_ratio
            },
            'statistics': self.stats,
            'output_path': str(self.split_dataset_path)
        }
        
        # Save report
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        report_path = self.outputs_path / f"classification_split_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Split report saved to: {report_path}")
        return report_path
    
    def verify_split_integrity(self):
        """
        Verify that the split was performed correctly.
        
        Returns:
            Boolean indicating if verification passed
        """
        print("\n" + "="*80)
        print("VERIFYING SPLIT INTEGRITY")
        print("="*80)
        
        errors = []
        
        # Check if PIL is available
        try:
            from PIL import Image
            pil_available = True
        except ImportError:
            pil_available = False
            print("\n⚠ Warning: PIL/Pillow not installed. Skipping image content verification.")
            print("  Install with: pip install Pillow\n")
        
        # Check each split directory
        for split in ['train', 'valid', 'test']:
            split_path = self.split_dataset_path / split
            
            if not split_path.exists():
                errors.append(f"Missing split directory: {split_path}")
                continue
            
            # Check each class
            classes_in_split = [d.name for d in split_path.iterdir() if d.is_dir()]
            
            for cls in classes_in_split:
                cls_path = split_path / cls
                images = self.get_image_files(cls_path)
                
                if not images:
                    errors.append(f"No images found in {cls_path}")
                
                # Verify all images are valid (only if PIL is available)
                if pil_available:
                    for img in images[:5]:  # Check first 5 images
                        img_path = cls_path / img
                        try:
                            with Image.open(img_path) as pil_img:
                                pil_img.verify()
                        except Exception as e:
                            errors.append(f"Corrupted image: {img_path} - {str(e)}")
        
        if errors:
            print(f"\n❌ Verification FAILED with {len(errors)} error(s):")
            for err in errors[:10]:  # Show first 10 errors
                print(f"  - {err}")
            return False
        else:
            print("\n✅ Verification PASSED - All splits are valid!")
            return True


def main():
    """Main function to run the dataset splitting."""
    print("\n🚀 Starting Image Classification Dataset Splitting...\n")
    
    # Initialize splitter
    splitter = ClassificationDatasetSplitter(
        student_id="25509225",
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15
    )
    
    try:
        # Perform splitting
        stats = splitter.perform_split()
        
        # Generate report
        report_path = splitter.generate_split_report()
        
        # Verify integrity
        is_valid = splitter.verify_split_integrity()
        
        if is_valid:
            print("\n" + "="*80)
            print("✅ DATASET SPLITTING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\nYou can now use the split dataset for training:")
            print(f"  Training:   {splitter.split_dataset_path}/train/")
            print(f"  Validation: {splitter.split_dataset_path}/valid/")
            print(f"  Testing:    {splitter.split_dataset_path}/test/")
            print("\nThe dataset is compatible with torchvision.datasets.ImageFolder")
            print("="*80)
        else:
            print("\n⚠ WARNING: Split verification failed. Please check the errors above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
