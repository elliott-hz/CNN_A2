**42028: Deep Learning and Convolutional Neural Network**

**Assignment \-2**

**Student Name: Kuanlong (Elliott) Li**

**Student ID: 25509225**

1. **Introduction:**

This report details the implementation, training, and evaluation of Convolutional Neural Networks (CNNs) for two core computer vision tasks: image classification and object detection. The primary focus is on demonstrating methodological rigor, ensuring fair experimental comparisons, and understanding architectural impacts on performance. 

For **image classification**, we employed the ResNet50 architecture as a baseline. To satisfy the customization requirements, we developed modified variants by altering the depth and structure of the convolutional backbone (e.g., adding or removing residual blocks). A critical constraint was that all layers remained fully trainable during fine-tuning to adhere to the assignment's "no freezing" policy.

For **object detection**, we implemented two distinct frameworks: **Faster R-CNN**, representing a two-stage region-based detector, and **YOLOv8**, a state-of-the-art single-stage real-time detector. These models were tasked with localizing and classifying various damage types in solar panel imagery. Throughout all experiments, consistent dataset splits based on Student ID `25509225` were utilized to ensure reproducibility and fairness across different model architectures.

2. **Dataset:**

Two specialized datasets were utilized for this assignment, both uniquely generated and assigned based on Student ID **25509225**.

**Image Classification Dataset (Bird Species):**
This dataset comprises **1,589 images** categorized into **10 distinct bird species**, including the Crested Kingfisher, Crow, Eastern Meadowlark, Fairy Bluebird, Harlequin Quail, Laughing Gull, Palila, Paradise Tanager, Rainbow Lorikeet, and Townsend's Warbler. Since the raw data was not pre-segregated, a stratified split was performed using the student ID as a random seed. The resulting distribution is:
*   **Training Set:** ~70% (1,109 images)
*   **Validation Set:** ~15% (231 images)
*   **Test Set:** ~15% (249 images)

*[Insert Figure 1: Sample images from each of the 10 bird classes]*

**Object Detection Dataset (Solar Panel Damage):**
This dataset contains **1,667 images** of solar panels, annotated with bounding boxes for **5 classes** of anomalies: Cell, Cell-Multi, No-Anomaly, Shadowing, and Unclassified. Unlike the classification dataset, this data was provided with pre-defined splits for training, validation, and testing. We utilized the COCO format annotations for the Faster R-CNN implementation and the YOLO format for the YOLOv8 framework to align with their respective data loading requirements.

*[Insert Figure 2: Sample images showing different types of solar panel damage and their corresponding bounding boxes]*

3. **Proposed CNN Architecture for Image Classification:**  
   1. Baseline architecture used  
      The baseline model is a standard **ResNet50** architecture. It consists of an initial convolutional layer followed by four residual blocks (layer1 to layer4) and a final fully connected classification head mapping the 2048-dimensional feature vector to the 10 bird species classes. All layers were trained from scratch without freezing any weights, using a learning rate of 5e-4 and a weight decay of 5e-4. The model incorporates Batch Normalization and ReLU activations throughout the backbone.

   2. Customized architecture  
      Two customized variants were developed to explore architectural impacts:
      *   **Deeper V1:** A convolutional block was added after `layer1` to increase the network's depth. This modification aims to enhance feature extraction capabilities in the early stages of processing. The total parameter count increased to approximately 24.7 million.
      *   **Reduced V1:** The entire `layer3` was removed from the ResNet50 backbone. This reduction significantly lowers the model's complexity (reducing parameters to ~15.1 million) and computational cost, testing whether a shallower network can maintain competitive performance on this specific dataset.

   3. Assumption/intuitions  
      The intuition behind **Deeper V1** is that increasing the receptive field and non-linearity through additional convolutional layers would allow the model to learn more complex patterns inherent in bird plumage and features. Conversely, **Reduced V1** is based on the assumption that the original ResNet50 might be over-parameterized for a 10-class problem with ~1,500 images. Removing `layer3` serves as a regularization technique to prevent overfitting and improve inference speed without sacrificing significant accuracy.

   4. Model Summary  

| Model Variant | Total Parameters | Trainable Parameters | Key Modification | Best Val Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (ResNet50)** | 23,528,522 | 23,528,522 | Standard Architecture | 95.24% | 93.17% |
| **Deeper V1** | 24,709,194 | 24,709,194 | Added Conv Block after Layer1 | 96.10% | 93.57% |
| **Reduced V1** | 15,119,434 | 15,119,434 | Removed Layer3 | 97.40% | 96.79% |

*[Insert Figure 3: Diagrams or summary tables of the three architectures]*

4. **CNN Architecture for Object Detection:**  
   1. Faster RCNN  
      \[Briefly explain the Faster-RCNN pipeline and the convolution base used (e.g., MobileNet, VGG16, etc.), add an image if possible.\]  
   2. SDD (Single Shot detector) or YOLO or detector of your choice  
      \[Briefly explain the pipeline and the convolution base used, add an image if possible.\]  
   3. Assumption/intuitions  
      \[Briefly explain the assumption made in the above two-object detection method\]  
   4. Model Summary  
      \[Provide the model summary of the two object detection methods\]  
        
5. **Experimental results and discussion:**  
   1. Experimental settings:   
      1. **Image Classification:**  
         \[Provide the hyper-parameter settings, data augmentation settings (if applied), transfer learning (if used, provide information on pre-trained model, etc.)\]  
      2. **Object Detection:**

      **\[**Provide the hyper-parameter settings, data augmentation settings (if applied), transfer learning (if used, provide information on pre-trained model, etc.)**\]**

   2. Experimental results:  
      1. **Image classification:**  
         1. **Performance on baseline/standard architecture**  
            **\[**Report the performance of the baseline CNN in a tabular format, with accuracy on train, validation, and test set. You may add train/validation/test loss as well.**\]**  
         2. **Performance on customized architecture**  
            **\[**Report the performance of the baseline CNN in a tabular format, with accuracy on train, validation, and test set. You may add train/validation/test loss as well.**\]**  
      2. **Object Detection:**  
         1. **Performance on Faster-RCNN**  
            **\[**Report the performance of Faster-RCNN in a tabular format, with mAP on train, validation, and test set. You may provide other details/analysis as well**\]**  
         2. **Performance on SDD/YOLO or any object detector**  
            **\[**Report the performance of SDD/Yolo/other object detector in a tabular format, with mAP on train, validation, and test set. You may provide other details/analysis as well.**\]**  
      3. **Discussion:**  
         **\[**Provide your understanding on the performance and accuracy obtained. Include some image classification and object detection result images. You may include wrongly classified/detected samples as well.**\]**

6. **Conclusion** 

\[Provide a short paragraph detailing your understanding on the experiments and results.\]