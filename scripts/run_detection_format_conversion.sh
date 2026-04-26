#!/bin/bash
# Run Detection Dataset Format Conversion
# Converts YOLO format to COCO JSON and VOC XML formats

set -e  # Exit on error

echo "=========================================="
echo "Detection Dataset Format Conversion"
echo "=========================================="
echo ""

# Check if source dataset exists
if [ ! -d "data/processed/detection" ]; then
    echo "Error: Source dataset not found at data/processed/detection"
    echo "Please run preprocessing first: bash scripts/run_data_preprocessing.sh"
    exit 1
fi

echo "Source dataset: data/processed/detection"
echo ""

# Run conversion script
echo "Converting to COCO and VOC formats..."
python src/data_processing/convert_detection_format.py --format both --source-dir data/processed/detection

echo ""
echo "=========================================="
echo "Conversion Complete!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - COCO format: data/processed/detection_coco/"
echo "  - VOC format:  data/processed/detection_voc/"
echo ""
echo "You can now run detection experiments:"
echo "  python experiments/exp01_detection_YOLOv8_baseline.py"
echo "  python experiments/exp02_detection_Faster-RCNN.py"
echo "  python experiments/exp03_detection_SSD.py"
echo ""
