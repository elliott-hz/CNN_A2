#!/bin/bash
# Run all 6 experiments sequentially

echo "=========================================="
echo "Running All Experiments"
echo "=========================================="

cd experiments

echo ""
echo "[1/6] Running Experiment 01: Detection Baseline..."
python exp01_detection_baseline.py

echo ""
echo "[2/6] Running Experiment 02: Detection Modified V1..."
python exp02_detection_modified_v1.py

echo ""
echo "[3/6] Running Experiment 03: Detection Modified V2..."
python exp03_detection_modified_v2.py

echo ""
echo "[4/6] Running Experiment 04: Classification Baseline..."
python exp04_classification_baseline.py

echo ""
echo "[5/6] Running Experiment 05: Classification Modified V1..."
python exp05_classification_modified_v1.py

echo ""
echo "[6/6] Running Experiment 06: Classification Modified V2..."
python exp06_classification_modified_v2.py

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
