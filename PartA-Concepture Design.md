# 42028: Deep Learning and Convolutional Neural Network Project Proposal
## Project Title
Visual Dog Emotion Recognition
---

## Abstract (~200 words)
This project aims to develop a Visual Dog Emotion Recognition system that can automatically identify dog emotions from facial expressions using computer vision and deep learning techniques. The system will classify emotions into five specific categories: angry, happy, relaxed, frown, and alert. A convolutional neural network (CNN) based model will be trained using a publicly available dataset of dog facial expressions containing approximately 10,000 images.

The system will first detect and process dog facial features from input images and then predict the corresponding emotional state.

After the model training process is completed, a simple web-based application will be developed. The system will include both front-end and back-end components and will be deployed on a cloud server. Users will be able to upload images or use a webcam to experience the real-time emotion recognition process online.

---

## Dataset Details
The project will primarily use the Dog Emotions Prediction and Recognition (5 classes) dataset.
- Total Images: 10,000+ images
- Image Type: RGB color images (standardized for canine facial features)
- 5 Emotion Categories:
  - Angry
  - Happy
  - Relaxed
  - Frown
  - Alert

The dataset is organized into training, test, and validation folders to facilitate robust model evaluation.

---

## Additional support required
1. **GPU resources**
A GPU-enabled environment will be required to accelerate the training process of deep learning models. Training convolutional neural networks on large image datasets can be computationally intensive, and GPU acceleration significantly improves training efficiency.

2. **Cloud server for deployment**
A standard cloud server will be used to deploy the trained model and host the application. The server will run a Python backend service that loads the trained model and performs emotion prediction. A simple web-based frontend interface will allow users to upload images or capture webcam input and receive real-time emotion predictions.