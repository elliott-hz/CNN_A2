#!/bin/bash
# Download datasets script

echo "=========================================="
echo "Downloading Datasets"
echo "=========================================="

python src/data_processing/download_datasets.py

echo ""
echo "Dataset download complete!"
echo "You can now run preprocessing or experiments."
