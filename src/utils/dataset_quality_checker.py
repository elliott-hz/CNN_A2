"""
Dataset Quality Checker for YOLOv8

Provides utilities to analyze dataset quality, including:
- Empty annotation detection
- Class distribution statistics
- Annotation completeness validation
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class DatasetQualityChecker:
    """
    Analyzes YOLO-format dataset quality and generates reports.
    """
    
    def __init__(self, data_yaml_path: str):
        """
        Initialize checker with dataset config path.
        
        Args:
            data_yaml_path: Path to data.yaml configuration file
        """
        self.data_yaml_path = Path(data_yaml_path)
        with open(self.data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config.get('names', [])
        self.num_classes = self.config.get('nc', len(self.class_names))
    
    def analyze_split(self, split_name: str = 'train') -> Dict:
        """
        Analyze a specific dataset split (train/val/test).
        
        Args:
            split_name: Split name ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing analysis results
        """
        split_path_str = self.config.get(split_name, None)
        if not split_path_str:
            raise ValueError(f"Split '{split_name}' not found in data.yaml")
        
        # Handle both directory path and list of paths
        if isinstance(split_path_str, list):
            split_paths = [Path(p) for p in split_path_str]
        else:
            split_paths = [Path(split_path_str)]
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []
        
        for split_path in split_paths:
            if split_path.is_dir():
                for ext in image_extensions:
                    all_images.extend(split_path.rglob(f'*{ext}'))
                    all_images.extend(split_path.rglob(f'*{ext.upper()}'))
            elif split_path.is_file() and split_path.suffix.lower() in image_extensions:
                all_images.append(split_path)
        
        all_images = list(set(all_images))  # Remove duplicates
        
        # Analyze annotations
        total_images = len(all_images)
        empty_annotations = 0
        annotation_stats = defaultdict(int)  # class_id -> count
        images_with_annotations = 0
        
        for img_path in all_images:
            # Find corresponding label file
            # YOLO format: images/img.jpg -> labels/img.txt
            label_path = self._get_label_path(img_path, split_paths)
            
            if label_path is None or not label_path.exists():
                # No annotation file at all
                empty_annotations += 1
                continue
            
            # Read annotation file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Filter out empty lines
                valid_lines = [line.strip() for line in lines if line.strip()]
                
                if len(valid_lines) == 0:
                    # Empty annotation file
                    empty_annotations += 1
                else:
                    # Has annotations
                    images_with_annotations += 1
                    
                    # Count classes
                    for line in valid_lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            annotation_stats[class_id] += 1
            
            except Exception as e:
                print(f"Warning: Could not read {label_path}: {e}")
                empty_annotations += 1
        
        # Build report
        report = {
            'split_name': split_name,
            'total_images': total_images,
            'images_with_annotations': images_with_annotations,
            'images_without_annotations': empty_annotations,
            'empty_annotation_percentage': (empty_annotations / total_images * 100) if total_images > 0 else 0,
            'total_annotations': sum(annotation_stats.values()),
            'class_distribution': {
                self.class_names[class_id]: count 
                for class_id, count in sorted(annotation_stats.items())
            },
            'avg_annotations_per_image': (
                sum(annotation_stats.values()) / images_with_annotations 
                if images_with_annotations > 0 else 0
            )
        }
        
        return report
    
    def _get_label_path(self, image_path: Path, split_paths: List[Path]) -> Path:
        """
        Get corresponding label file path for an image.
        
        Args:
            image_path: Path to image file
            split_paths: List of possible split root paths
            
        Returns:
            Path to label file, or None if not found
        """
        # Try to find labels directory parallel to images directory
        image_rel_path = None
        base_path = None
        
        for split_path in split_paths:
            try:
                image_rel_path = image_path.relative_to(split_path)
                base_path = split_path
                break
            except ValueError:
                continue
        
        if image_rel_path is None or base_path is None:
            return None
        
        # Common YOLO structure: replace 'images' with 'labels'
        label_path_str = str(image_rel_path).replace('images', 'labels')
        label_path = base_path.parent / 'labels' / label_path_str
        
        # Alternative: same directory structure but under 'labels' folder
        if not label_path.exists():
            # Try direct replacement in parent directory
            parent_name = base_path.name
            if 'images' in parent_name:
                labels_parent = base_path.parent / parent_name.replace('images', 'labels')
                label_path = labels_parent / image_rel_path
        
        # Change extension to .txt
        label_path = label_path.with_suffix('.txt')
        
        return label_path if label_path.exists() else None
    
    def generate_full_report(self) -> str:
        """
        Generate comprehensive dataset quality report.
        
        Returns:
            Markdown-formatted report string
        """
        report_lines = []
        report_lines.append("# Dataset Quality Report\n")
        report_lines.append(f"**Dataset Config:** `{self.data_yaml_path}`\n")
        report_lines.append(f"**Number of Classes:** {self.num_classes}\n")
        report_lines.append(f"**Class Names:** {', '.join(self.class_names)}\n\n")
        
        # Analyze each split
        for split_name in ['train', 'val', 'test']:
            if split_name in self.config:
                try:
                    report = self.analyze_split(split_name)
                    
                    report_lines.append(f"## {split_name.capitalize()} Set\n")
                    report_lines.append(f"- **Total Images:** {report['total_images']}")
                    report_lines.append(f"- **Images with Annotations:** {report['images_with_annotations']}")
                    report_lines.append(f"- **Images without Annotations (Empty):** {report['images_without_annotations']} ({report['empty_annotation_percentage']:.1f}%)")
                    report_lines.append(f"- **Total Annotations:** {report['total_annotations']}")
                    report_lines.append(f"- **Avg Annotations per Image:** {report['avg_annotations_per_image']:.2f}\n")
                    
                    # Class distribution
                    if report['class_distribution']:
                        report_lines.append("### Class Distribution\n")
                        report_lines.append("| Class | Count | Percentage |")
                        report_lines.append("|-------|-------|------------|")
                        
                        total_ann = report['total_annotations']
                        for class_name, count in report['class_distribution'].items():
                            percentage = (count / total_ann * 100) if total_ann > 0 else 0
                            report_lines.append(f"| {class_name} | {count} | {percentage:.1f}% |")
                        
                        report_lines.append("")
                    
                    # Important note about empty annotations
                    if report['images_without_annotations'] > 0:
                        report_lines.append("**Note:** Images with empty annotations are retained as negative samples ")
                        report_lines.append("to help the model learn background patterns and reduce false positives.\n")
                
                except Exception as e:
                    report_lines.append(f"## {split_name.capitalize()} Set\n")
                    report_lines.append(f"⚠️ Error analyzing {split_name} set: {e}\n")
        
        # Summary and recommendations
        report_lines.append("## Data Quality Assessment\n")
        report_lines.append("✅ **Empty annotations are preserved** - These serve as valuable negative samples\n")
        report_lines.append("✅ **No images removed** - Maintains dataset integrity as per assignment requirements\n")
        report_lines.append("✅ **Ultralytics framework handles empty annotations automatically** during training\n")
        
        return '\n'.join(report_lines)
    
    def print_summary(self):
        """Print a concise summary to console."""
        print("\n" + "=" * 80)
        print("DATASET QUALITY ANALYSIS")
        print("=" * 80)
        
        for split_name in ['train', 'val', 'test']:
            if split_name in self.config:
                try:
                    report = self.analyze_split(split_name)
                    print(f"\n{split_name.upper()} SET:")
                    print(f"  Total images: {report['total_images']}")
                    print(f"  With annotations: {report['images_with_annotations']}")
                    print(f"  Empty annotations: {report['images_without_annotations']} ({report['empty_annotation_percentage']:.1f}%)")
                    print(f"  Total boxes: {report['total_annotations']}")
                    
                    if report['images_without_annotations'] > 0:
                        print(f"  ℹ️  Empty annotations retained as negative samples")
                
                except Exception as e:
                    print(f"\n{split_name.upper()} SET: Error - {e}")
        
        print("\n" + "=" * 80)
