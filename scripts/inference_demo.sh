#!/bin/bash
# Inference demo script

if [ -z "$1" ]; then
    echo "Usage: ./inference_demo.sh <image_path>"
    echo "Example: ./inference_demo.sh test_image.jpg"
    exit 1
fi

IMAGE_PATH=$1

echo "=========================================="
echo "Running Inference Pipeline Demo"
echo "=========================================="

# Find latest model paths (you may need to adjust these)
DETECTION_MODEL="outputs/exp01_detection_baseline/run_*/model/best_model.pt"
CLASSIFICATION_MODEL="outputs/exp04_classification_baseline/run_*/model/best_model.pth"

python -c "
from src.inference.pipeline_inference import PipelineInference
import glob

# Find latest models
detection_models = sorted(glob.glob('$DETECTION_MODEL'))
classification_models = sorted(glob.glob('$CLASSIFICATION_MODEL'))

if not detection_models or not classification_models:
    print('Error: Model files not found. Please run experiments first.')
    exit(1)

detection_model = detection_models[-1]
classification_model = classification_models[-1]

print(f'Using detection model: {detection_model}')
print(f'Using classification model: {classification_model}')

# Initialize pipeline
pipeline = PipelineInference(detection_model, classification_model)

# Run inference
results = pipeline.predict('$IMAGE_PATH')

# Print results
print('\nDetection Results:')
for result in results:
    print(f\"  Dog {result['dog_id']}:\")
    print(f\"    Emotion: {result['emotion']} (confidence: {result['emotion_confidence']:.2f})\")
    print(f\"    Detection confidence: {result['detection_confidence']:.2f}\")

# Visualize
output_path = 'inference_output.jpg'
pipeline.visualize('$IMAGE_PATH', output_path)
print(f'\nVisualization saved to: {output_path}')
"
