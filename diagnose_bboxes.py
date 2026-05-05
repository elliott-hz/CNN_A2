"""
Diagnostic script to check for invalid bounding boxes in the dataset.
This helps identify dataset issues before training.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def diagnose_coco_bboxes(data_root):
    """Diagnose COCO format annotations for invalid bboxes."""
    print("\n" + "="*80)
    print("DIAGNOSING COCO FORMAT ANNOTATIONS")
    print("="*80)
    
    data_root = Path(data_root)
    splits = ['train', 'valid', 'test']
    
    total_stats = defaultdict(int)
    
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"\n⚠️  {split}/ directory not found")
            continue
        
        # Find JSON annotation file
        json_files = list(split_dir.glob('*.json'))
        if not json_files:
            print(f"\n⚠️  No JSON annotation files found in {split}/")
            continue
        
        annot_file = json_files[0]
        print(f"\n[{split.upper()}] Analyzing {annot_file.name}")
        
        with open(annot_file, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        
        print(f"  Total images: {len(images)}")
        print(f"  Total annotations: {len(annotations)}")
        
        invalid_count = 0
        invalid_details = []
        
        for idx, annot in enumerate(annotations):
            x, y, w, h = annot.get('bbox', [0, 0, 0, 0])
            
            if w <= 0 or h <= 0:
                invalid_count += 1
                if len(invalid_details) < 10:  # Show first 10 invalid boxes
                    invalid_details.append({
                        'annotation_id': annot.get('id'),
                        'image_id': annot.get('image_id'),
                        'bbox': [x, y, w, h],
                        'area': annot.get('area')
                    })
        
        total_stats[f'{split}_invalid'] += invalid_count
        total_stats[f'{split}_total'] += len(annotations)
        
        if invalid_count > 0:
            print(f"\n  ❌ Found {invalid_count} invalid bboxes (width ≤ 0 or height ≤ 0)")
            print(f"     Invalid ratio: {invalid_count / len(annotations) * 100:.2f}%")
            
            if invalid_details:
                print(f"\n  First few invalid annotations:")
                for detail in invalid_details:
                    print(f"    - Annotation ID {detail['annotation_id']}: bbox={detail['bbox']}, area={detail['area']}")
        else:
            print(f"  ✅ All {len(annotations)} bboxes are valid (w > 0, h > 0)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for split in splits:
        invalid = total_stats.get(f'{split}_invalid', 0)
        total = total_stats.get(f'{split}_total', 0)
        if total > 0:
            ratio = invalid / total * 100
            status = "❌" if invalid > 0 else "✅"
            print(f"{status} {split:8s}: {invalid:4d}/{total:4d} invalid bboxes ({ratio:5.2f}%)")


def main():
    """Main diagnostic function."""
    print("\n" + "="*80)
    print("BOUNDING BOX VALIDATION DIAGNOSTIC")
    print("="*80)
    
    student_id = "25509225"
    data_root = f"data/{student_id}/Object_Detection/coco"
    
    if not Path(data_root).exists():
        print(f"❌ Data directory not found: {data_root}")
        sys.exit(1)
    
    diagnose_coco_bboxes(data_root)
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("The dataloader now has validation to skip invalid bboxes.")
    print("Images with all invalid bboxes will be treated as having no objects.")
    print("This allows training to continue despite dataset imperfections.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
