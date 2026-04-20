#!/bin/bash
# Run a single experiment by number

if [ -z "$1" ]; then
    echo "Usage: ./run_single_experiment.sh <experiment_number>"
    echo "Example: ./run_single_experiment.sh 01"
    exit 1
fi

EXP_NUM=$1
EXP_FILE="experiments/exp${EXP_NUM}_*.py"

echo "=========================================="
echo "Running Experiment ${EXP_NUM}"
echo "=========================================="

cd experiments
python exp${EXP_NUM}_*.py

echo ""
echo "Experiment ${EXP_NUM} completed!"
